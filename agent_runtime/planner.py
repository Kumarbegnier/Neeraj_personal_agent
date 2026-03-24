from __future__ import annotations

from src.schemas.planner import PlannerOutput, PlannerSubtask
from src.services.planner_service import PlannerService
from src.services.llm_service import LLMService

from .models import AgentState, ExecutionPlan, PlanStep, ReActCycle
from .runtime_utils import dedupe_preserve_order, record_model_result, selected_model_override


class Planner:
    def __init__(
        self,
        llm_service: LLMService | None = None,
        planner_service: PlannerService | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._planner_service = planner_service or (
            PlannerService(llm_service) if llm_service is not None else None
        )

    def create_plan(self, state: AgentState) -> ExecutionPlan:
        fallback = self._fallback_plan(state)
        if self._planner_service is None:
            return fallback

        context = state.context
        control = state.control
        planner_result, critique_result = self._planner_service.generate_plan(
            objective=state.request.message,
            intent=control.intent if control else "unknown",
            complexity=control.complexity if control else "unknown",
            requested_capabilities=context.requested_capabilities if context else [],
            constraints=self._combined_constraints(state),
            verification_focus=self._verification_focus(state),
            fallback_output=self._planner_output_from_plan(fallback),
            selected_model=selected_model_override(state),
            metadata={"step_index": state.step_index, "retry_count": state.retry_count},
        )
        record_model_result(state, planner_result)
        record_model_result(state, critique_result)
        planner_output = PlannerOutput.model_validate(planner_result.output.model_dump())
        return self._merge_planner_output(fallback, planner_output)

    def _fallback_plan(self, state: AgentState) -> ExecutionPlan:
        context = state.context
        control = state.control
        if context is None or control is None:
            raise ValueError("Context and control must exist before planning.")

        assumptions = self._assumptions(state)
        success_criteria = self._success_criteria(state)
        failure_modes = self._failure_modes(state)
        verification_focus = self._verification_focus(state)
        decomposition_strategy = self._decomposition_strategy(state)

        react_cycles = [
            ReActCycle(
                thought="Observe the latest user objective, retrieved memory, and prior attempt outcomes.",
                action="Refresh working memory and identify what changed since the previous step.",
                observation=(
                    f"Loaded {len(context.memory.retrieved)} retrieved memories and "
                    f"{len(state.observations)} accumulated observations."
                ),
            ),
            ReActCycle(
                thought="Control the loop from the current state rather than replaying a fixed pipeline.",
                action="Choose or revise the control posture, specialist route, and verification mode.",
                observation=(
                    f"Preferred specialist is '{control.preferred_agent or 'general'}' with "
                    f"replan_count={state.replan_count} and retry_count={state.retry_count}."
                ),
            ),
            ReActCycle(
                thought="Preserve a closed loop by making failure information causally binding.",
                action="Convert verification gaps and reflection repairs into constraints for the next action.",
                observation=(
                    f"Adaptive constraints count is {len(state.adaptive_constraints)} and "
                    f"blocked_tools count is {len(state.blocked_tools)}."
                ),
            ),
        ]

        steps = [
            PlanStep(
                name="observe",
                description="Refresh context, retrieved memory, and working memory from the live AgentState.",
                owner="context_builder",
                status="completed",
                step_type="memory",
                success_criteria=["Relevant memory and recent observations are loaded into working memory."],
                verification_focus=["Observation incorporates current memory and loop feedback."],
                requires_tools=["semantic_memory", "vector_memory", "working_memory"],
            ),
            PlanStep(
                name="control",
                description="Recompute control posture, intent, and specialist preference from the current AgentState.",
                owner="orchestrator_brain",
                status="completed",
                step_type="control",
                success_criteria=["Control policy reflects current memory, retry state, and reflection feedback."],
                failure_modes=["The system repeats a failed strategy without changing constraints."],
                verification_focus=["State changes are causally reflected in control decisions."],
            ),
            PlanStep(
                name="plan",
                description="Generate or re-generate the execution plan from current control state and loop feedback.",
                owner="planner",
                status="completed",
                step_type="planning",
                depends_on=["observe", "control"],
                success_criteria=["The plan reflects retry, replan, and reflection feedback."],
                failure_modes=["The plan reuses stale assumptions after verification failure."],
                verification_focus=["State changes are causally reflected in the current plan."],
            ),
            PlanStep(
                name="route",
                description="Select the next specialist route using message features, memory bias, and reflection hints.",
                owner="router_executor",
                status="in_progress",
                step_type="routing",
                depends_on=["observe", "control", "plan"],
                success_criteria=["The route is informed by current goals, retrieved memory, and reflection bias."],
                verification_focus=["Routing rationale changes when state changes."],
            ),
            PlanStep(
                name="agent_decide",
                description="Let the routed specialist convert the current state into a tool-backed action decision.",
                owner="router_executor",
                depends_on=["route"],
                step_type="decision",
                success_criteria=["Tool selection reflects current route, constraints, and blocked tools."],
                failure_modes=["The agent chooses tools that ignore reflection repairs or blocked-tool constraints."],
                verification_focus=["Agent decision changes when route bias, memory, or constraints change."],
            ),
            PlanStep(
                name="execute",
                description="Execute the chosen tool set, collect observations, and update the AgentState.",
                owner="execution",
                depends_on=["agent_decide"],
                step_type="execution",
                success_criteria=["Actions create observable evidence or explicit gates."],
                failure_modes=["Execution repeats blocked tools or yields no new observations."],
                verification_focus=["Tool outputs and observations are captured in state."],
                requires_tools=context.requested_capabilities or ["reasoning"],
                risk_level=control.risk_level,
            ),
            PlanStep(
                name="verify",
                description="Verify that recent claims and progress are grounded in evidence.",
                owner="verification",
                depends_on=["execute"],
                step_type="verification",
                success_criteria=["Failed verification leads to retry or replan."],
                failure_modes=["Verification is ignored and execution continues unchanged."],
                verification_focus=verification_focus,
            ),
            PlanStep(
                name="reflect",
                description="Translate verification failures into state mutations that change future behavior.",
                owner="reflection",
                depends_on=["verify"],
                step_type="reflection",
                success_criteria=["Reflection mutates constraints, route bias, or blocked tools when needed."],
                verification_focus=["Reflection is causal, not decorative."],
            ),
            PlanStep(
                name="update_state",
                description="Apply the explicit transition S_{t+1} = F(S_t, O_t) by checkpointing memory and updated control state.",
                owner="orchestrator",
                depends_on=["reflect"],
                step_type="state_transition",
                success_criteria=["Observations and repairs are written into the next loop state."],
                verification_focus=["State updates causally carry observations into the next iteration."],
            ),
            PlanStep(
                name="finalize_response",
                description="Compose the user-facing response only after the loop finishes.",
                owner="response",
                depends_on=["update_state"],
                step_type="response",
                success_criteria=success_criteria,
                verification_focus=["The response is synthesized from final state, not intermediate templates."],
            ),
        ]

        reasoning = (
            f"Step {state.step_index}: built a {decomposition_strategy} plan for intent '{control.intent}' using "
            f"{len(context.history)} recent turns, {len(context.memory.retrieved)} retrieved memories, and "
            f"{len(state.observations)} accumulated observations."
        )

        return ExecutionPlan(
            objective=state.request.message.strip(),
            task_summary=(
                f"Build a {decomposition_strategy} plan for intent '{control.intent}' "
                f"using current memory, verification focus, and loop feedback."
            ),
            subtasks=self._planner_subtasks(steps),
            required_tools=dedupe_preserve_order(context.requested_capabilities or ["reasoning"]),
            risk_level=control.risk_level,
            approval_needed=control.risk_level in {"high", "critical"},
            reasoning=reasoning,
            react_cycles=react_cycles,
            steps=steps,
            assumptions=assumptions,
            constraints=self._combined_constraints(state),
            success_criteria=success_criteria,
            failure_modes=failure_modes,
            verification_focus=verification_focus,
            decomposition_strategy=decomposition_strategy,
            completion_state="replanned" if state.needs_replan else "planned",
        )

    def complete(self, plan: ExecutionPlan) -> ExecutionPlan:
        completed_steps = [step.model_copy(update={"status": "completed"}) for step in plan.steps]
        return plan.model_copy(update={"steps": completed_steps, "completion_state": "completed"})

    def _assumptions(self, state: AgentState) -> list[str]:
        assumptions = [
            "The runtime should operate as a closed-loop agent rather than a one-pass pipeline.",
            "The final response must be generated only after the loop converges or hits its limit.",
        ]
        if state.memory.retrieved:
            assumptions.append("Retrieved memory is binding evidence for routing and reasoning bias.")
        if state.control and state.control.preferred_agent:
            assumptions.append(f"The current best specialist is '{state.control.preferred_agent}'.")
        if state.reflection and state.reflection.lessons:
            assumptions.extend(state.reflection.lessons[:2])
        return dedupe_preserve_order(assumptions)

    def _success_criteria(self, state: AgentState) -> list[str]:
        criteria = [
            "The loop reaches a verified or explicitly bounded stopping condition.",
            "Memory, verification, and reflection causally influence later decisions.",
            "The final response is synthesized from the converged AgentState.",
        ]
        if state.retry_count > 0:
            criteria.append("A failed attempt causes a changed strategy rather than a repeated one.")
        return criteria

    def _failure_modes(self, state: AgentState) -> list[str]:
        modes = [
            "The planner ignores prior verification or reflection outcomes.",
            "The runtime repeats blocked tools or routes without changing state.",
            "Intermediate agent summaries leak directly as the final user response.",
        ]
        if state.blocked_tools:
            modes.append("Blocked tools remain in the candidate tool set.")
        return modes

    def _verification_focus(self, state: AgentState) -> list[str]:
        focus = [
            "Recent observations provide evidence for the claimed progress.",
            "Retrieved memory is relevant to the chosen route and action.",
            "Verification failures change the next plan or action.",
        ]
        if state.route_bias:
            focus.append("Reflection-provided route bias is actually honored.")
        if state.blocked_tools:
            focus.append("Blocked tools are excluded from the next action set.")
        return focus

    def _decomposition_strategy(self, state: AgentState) -> str:
        if state.retry_count > 0 or state.needs_replan:
            return "adaptive retry loop"
        if state.context and state.context.signals.collaboration_mode == "modular-orchestration":
            return "hierarchical modular"
        if state.control and state.control.complexity == "high":
            return "deliberate multistep"
        return "compact specialist"

    def _combined_constraints(self, state: AgentState) -> list[str]:
        context_constraints = state.context.constraints if state.context else []
        return dedupe_preserve_order([*context_constraints, *state.adaptive_constraints])

    def _planner_subtasks(self, steps: list[PlanStep]) -> list[PlannerSubtask]:
        return [
            PlannerSubtask(
                name=step.name,
                description=step.description,
                owner=step.owner,
                depends_on=step.depends_on,
                required_tools=step.requires_tools,
                expected_output=step.success_criteria[0] if step.success_criteria else "",
                risk_level=step.risk_level,
            )
            for step in steps
        ]

    def _planner_output_from_plan(self, plan: ExecutionPlan) -> PlannerOutput:
        return PlannerOutput(
            task_summary=plan.task_summary,
            subtasks=plan.subtasks or self._planner_subtasks(plan.steps),
            required_tools=plan.required_tools,
            risk_level=plan.risk_level,
            approval_needed=plan.approval_needed,
            success_criteria=plan.success_criteria,
            assumptions=plan.assumptions,
            constraints=plan.constraints,
            failure_modes=plan.failure_modes,
            verification_focus=plan.verification_focus,
            decomposition_strategy=plan.decomposition_strategy,
            reasoning=plan.reasoning,
        )

    def _merge_planner_output(
        self,
        fallback: ExecutionPlan,
        planner_output: PlannerOutput,
    ) -> ExecutionPlan:
        merged_steps = self._merge_steps(
            fallback.steps,
            planner_output.subtasks,
            planner_output.required_tools,
            planner_output.risk_level,
        )
        return fallback.model_copy(
            update={
                "task_summary": planner_output.task_summary or fallback.task_summary,
                "subtasks": planner_output.subtasks or fallback.subtasks,
                "required_tools": planner_output.required_tools or fallback.required_tools,
                "risk_level": planner_output.risk_level or fallback.risk_level,
                "approval_needed": planner_output.approval_needed,
                "success_criteria": planner_output.success_criteria or fallback.success_criteria,
                "assumptions": planner_output.assumptions or fallback.assumptions,
                "constraints": planner_output.constraints or fallback.constraints,
                "failure_modes": planner_output.failure_modes or fallback.failure_modes,
                "verification_focus": planner_output.verification_focus or fallback.verification_focus,
                "decomposition_strategy": (
                    planner_output.decomposition_strategy or fallback.decomposition_strategy
                ),
                "reasoning": planner_output.reasoning or fallback.reasoning,
                "steps": merged_steps,
            }
        )

    def _merge_steps(
        self,
        fallback_steps: list[PlanStep],
        subtasks: list[PlannerSubtask],
        required_tools: list[str],
        default_risk_level: str,
    ) -> list[PlanStep]:
        subtask_map = {subtask.name: subtask for subtask in subtasks}
        merged_steps: list[PlanStep] = []

        for step in fallback_steps:
            matched = subtask_map.get(step.name)
            if matched is None:
                if step.name == "execute" and required_tools:
                    merged_steps.append(
                        step.model_copy(
                            update={
                                "requires_tools": required_tools,
                                "risk_level": default_risk_level or step.risk_level,
                            }
                        )
                    )
                else:
                    merged_steps.append(step)
                continue

            merged_steps.append(
                step.model_copy(
                    update={
                        "description": matched.description or step.description,
                        "owner": matched.owner or step.owner,
                        "depends_on": matched.depends_on or step.depends_on,
                        "requires_tools": matched.required_tools or step.requires_tools,
                        "risk_level": matched.risk_level or step.risk_level or default_risk_level,
                    }
                )
            )

        existing_names = {step.name for step in merged_steps}
        for subtask in subtasks:
            if subtask.name in existing_names:
                continue
            merged_steps.append(
                PlanStep(
                    name=subtask.name,
                    description=subtask.description,
                    owner=subtask.owner,
                    depends_on=subtask.depends_on,
                    step_type="planning",
                    requires_tools=subtask.required_tools,
                    success_criteria=[subtask.expected_output] if subtask.expected_output else [],
                    risk_level=subtask.risk_level or default_risk_level,
                )
            )

        return merged_steps
