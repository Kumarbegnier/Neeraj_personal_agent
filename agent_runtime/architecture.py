from pydantic import BaseModel


class StageDescriptor(BaseModel):
    name: str
    component: str
    description: str


ARCHITECTURE_STAGES = [
    StageDescriptor(
        name="User / UI",
        component="Voice / Text / API / UI",
        description="Accepts user requests from interface clients and hands them to the stateful runtime.",
    ),
    StageDescriptor(
        name="API Gateway",
        component="FastAPI / Auth / Rate",
        description="Normalizes inbound requests and enforces authentication and rate policies.",
    ),
    StageDescriptor(
        name="Session + Permissions",
        component="SessionPermissionManager",
        description="Restores session state and determines whether execution is auto-approved, confirm-required, or blocked.",
    ),
    StageDescriptor(
        name="Initialize AgentState",
        component="AgentState",
        description="Creates the single evolving runtime state object that persists across loop steps.",
    ),
    StageDescriptor(
        name="Observe",
        component="ContextBuilder + MemorySystem",
        description="Refreshes context, retrieved memory, goals, constraints, and working memory onto the live AgentState.",
    ),
    StageDescriptor(
        name="Select Architecture",
        component="ArchitectureSelector",
        description="Chooses the execution pattern that best fits the task's complexity, grounding needs, tool intensity, risk, and parallelism.",
    ),
    StageDescriptor(
        name="Control",
        component="OrchestratorBrain",
        description="Computes control posture from the current AgentState rather than from the initial request alone.",
    ),
    StageDescriptor(
        name="Plan",
        component="Planner",
        description="Generates or regenerates the execution plan, including retry-driven replanning and verification focus.",
    ),
    StageDescriptor(
        name="Route",
        component="AgentRouter + RouterExecutor",
        description="Selects the next specialist lane using current goals, retrieved memory, and reflection-derived route bias.",
    ),
    StageDescriptor(
        name="Agent Decide",
        component="Specialized Agents + RouterExecutor",
        description="Lets the routed specialist translate the current AgentState into a structured tool-backed action decision.",
    ),
    StageDescriptor(
        name="Execute",
        component="ExecutionEngine + ToolRegistry",
        description="Executes the selected tool calls and records observations and execution summaries back into AgentState.",
    ),
    StageDescriptor(
        name="Verify",
        component="VerificationEngine",
        description="Checks whether tool execution, observations, and claims are sufficiently grounded and can force retry.",
    ),
    StageDescriptor(
        name="Reflect",
        component="ReflectionEngine",
        description="Turns verification failures into route bias, blocked-tool constraints, and adaptive repairs that change future steps.",
    ),
    StageDescriptor(
        name="Update State",
        component="MemorySystem + AgentState",
        description="Applies the explicit state transition S_{t+1} = F(S_t, O_t) and checkpoints the next live state.",
    ),
    StageDescriptor(
        name="Safety Review",
        component="SafetyPermissions",
        description="Applies policy checks and confirmation gates after the loop converges on a candidate result.",
    ),
    StageDescriptor(
        name="Final Response Synthesis",
        component="ResponseComposer",
        description="Generates the user-facing response from final AgentState only after verification, reflection, and safety settle.",
    ),
    StageDescriptor(
        name="Persistent Memory Update",
        component="MemorySystem",
        description="Persists the conversation turn together with route choice, verification summaries, and reflection lessons.",
    ),
    StageDescriptor(
        name="Response / Approval / Action Result",
        component="InteractionResponse",
        description="Returns the final answer together with approval requirements, loop count, verification state, and trace.",
    ),
]
