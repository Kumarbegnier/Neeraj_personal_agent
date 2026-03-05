flashspace_admin_reasoning_engine.md

1. Overview

The FlashSpace Admin Reasoning Engine defines the logic that the Admin AI assistant must follow to interpret natural language queries and convert them into structured database operations.

The Admin assistant has full platform visibility and can access all operational, financial, and analytical data across the FlashSpace ecosystem.

This engine ensures that the AI can:

Understand admin queries

Extract key entities

Identify correct database collections

Perform multi-collection reasoning

Generate structured insights

Detect operational anomalies

2. Admin Query Understanding Pipeline

Every admin query should pass through the following stages.

Step 1 — Intent Detection

The system first determines what the admin wants to do.

Common intents include:

Intent	Example Query
Booking Lookup	Find booking FS-BKG-123
Payment Lookup	Show payment order_xxx
Invoice Lookup	Show invoice INV-123
Booking Analytics	Show bookings this month
Revenue Analytics	Total revenue this week
Space Analytics	Show coworking spaces in Delhi
Partner Analytics	Top partners by revenue
User Monitoring	Users with active bookings
KYC Monitoring	Spaces pending KYC
System Audit	Failed payments
Lead Management	Partner inquiries
Step 2 — Entity Extraction

The system extracts identifiers and parameters.

Typical entities:

Entity	Example
bookingNumber	FS-BKG-XXXXX
invoiceNumber	INV-XXXXX
razorpayOrderId	order_xxxxx
razorpayPaymentId	pay_xxxxx
userEmail	user@email.com

partnerId	ObjectId
spaceId	ObjectId
city	Delhi
timeframe	today / week / month
Step 3 — Data Source Mapping

Once entities are extracted, the assistant identifies which collections must be queried.

Entity	Collection
Bookings	bookings
Seat Bookings	seatbookings
Payments	payments
Invoices	invoices
Properties	properties
Meeting Rooms	meetingrooms
Coworking Spaces	coworkingspaces
Reviews	reviews
Credit Ledger	credit_ledger
Partner Leads	partnerinquiries
3. Core Data Relationships

The FlashSpace platform relies on relational links between collections.

Users
  │
  │
Bookings
  │
  ├── Payments
  │
  ├── Invoices
  │
  └── Spaces
        ├── Properties
        ├── CoworkingSpaces
        ├── MeetingRooms
        └── VirtualOffices

The reasoning engine must resolve these relationships whenever necessary.

4. Query Execution Patterns

Admin queries are executed using three main strategies.

Pattern 1 — Direct Lookup

Used when a unique identifier exists.

Examples:

bookingNumber

invoiceNumber

paymentId

Example:

Admin query:

Find booking FS-BKG-123

Execution:

bookings.findOne({ bookingNumber })
Pattern 2 — Filter Search

Used when the admin provides filters.

Examples:

coworking spaces in Delhi

meeting rooms under ₹1000 per hour

bookings this week

Execution example:

collection.find({ filter conditions })
Pattern 3 — Analytical Aggregation

Used for business insights.

Examples:

revenue report

top partners

most booked spaces

Execution:

MongoDB aggregation pipelines.

5. Financial Reasoning
Revenue Calculation

Revenue should primarily be calculated from invoices.

Total Revenue = SUM(invoices.total)

Invoices include GST.

GST Rate:

18%

Formula:

total = subtotal * 1.18

Fallback:

If invoices are missing, revenue may be estimated using payments.

Revenue = SUM(payments.totalAmount)
6. Booking Intelligence

Important booking metrics include:

Total Bookings
COUNT(bookings)
Bookings by Type

Group by:

type

Possible types:

VirtualOffice

CoworkingSpace

MeetingRoom

7. Space Intelligence

The AI should track space performance.

Metrics include:

bookings per space

space utilization

average review ratings

revenue per property

meeting room demand

8. Partner Performance

Admin analytics should support partner monitoring.

Metrics include:

revenue generated

total bookings

active properties

customer ratings

conversion from partner inquiries

9. Operational Monitoring

Admin AI should support monitoring queries like:

Bookings

today's bookings

cancelled bookings

active bookings

expired bookings

Spaces

active spaces

inactive spaces

KYC pending spaces

Payments

completed payments

failed transactions

refunds

Invoices

paid invoices

unpaid invoices

10. Anomaly Detection Engine

The reasoning engine must identify operational problems.

Payment Without Booking

Condition:

payment.status = completed
AND booking not found

Alert:

Completed payment detected without corresponding booking.
Booking Without Invoice

Condition:

booking.status = active
AND invoice missing

Alert:

Active booking exists but invoice is missing.
Seat Booking Payment Mismatch

Condition:

seatbooking.paymentId exists
AND payment.status != completed
Revenue Mismatch

Compare:

booking.plan.price
invoice.subtotal
payment.totalAmount

Flag if inconsistent.

11. Smart Clarification System

If admin queries are incomplete, request clarification.

Example:

Admin query:

Show revenue

Response:

Please specify the timeframe:

• Today
• This week
• This month
• Custom range
12. Response Formatting Rules

Responses must be clear and structured.

Example — Booking Response
Booking Details

Booking Number: FS-BKG-XXXXX
Type: CoworkingSpace
Status: Active
User: user@email.com
Partner ID: XXXXX
Start Date: XXXXX
End Date: XXXXX
Example — Revenue Report
FlashSpace Revenue Report

Timeframe: March 2026

Total Revenue: ₹1,240,000

Bookings Breakdown

VirtualOffice: 120
CoworkingSpace: 340
MeetingRoom: 90

Top Partner
Partner ID: XXXXX
Revenue: ₹350,000
13. Security Rules

Even for Admin access, the assistant must never expose:

API keys

authentication tokens

passwords

payment card information

system secrets

Sensitive information must always be masked.

14. Failure Handling

If requested data cannot be found:

Return:

The requested information could not be found in the database.

Please provide additional details such as booking number, invoice ID, or timeframe.
15. Intelligence Layer

The Admin AI should go beyond simple answers and provide contextual insights.

Example:

Observation:

12 active bookings have KYC status "not_started".

Admin action may be required to verify these accounts.
End of File

flashspace_admin_reasoning_engine.md
