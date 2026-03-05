# flashspace_admin_query_logic.md

## 1) Purpose
This document defines how the FlashSpace **Admin AI** should convert an admin question into:
- the right **collections to query**
- the right **filters & joins**
- a clear, **professional report-style answer**

Admin can access platform-wide data (bookings, payments, invoices, spaces, partners, reviews, seat bookings, credits, partner leads).

---

## 2) Collections & What They Represent

### Operational
- `bookings` -> primary booking lifecycle (VirtualOffice / CoworkingSpace / MeetingRoom)  
- `seatbookings` -> short-term seat reservations (often with `paymentId` string)

### Commerce
- `payments` -> transaction records (Razorpay Order/Payment IDs, amounts, status, paymentType, spaceModel)
- `invoices` -> billing records (invoiceNumber, subtotal, taxRate, total, status, user, partner)

### Supply / Inventory
- `properties` -> base property listing (name, address, city, area, kycStatus, partner, isActive)
- `meetingrooms` -> meeting room configuration + pricing for a property
- `coworkingspaces` -> coworking inventory (floors, tables, seats) + monthly pricing

### Trust / Ops
- `reviews` -> ratings by user for a space (spaceModel + space ObjectId)
- `credit_ledger` -> user credits / bonuses / balances
- `partnerinquiries` -> partner lead submissions (status pending etc.)

---

## 3) Key IDs & Fields Admin Should Recognize

### Booking Keys
- `bookingNumber` (e.g., FS-BKG-...)  
- `type` (VirtualOffice / CoworkingSpace / MeetingRoom)  
- `user`, `partner`, `spaceId`, `spaceSnapshot.name`, `status`, `kycStatus`, `startDate`, `endDate`

### Payment Keys
- `razorpayOrderId`, `razorpayPaymentId`
- `status` (completed etc.)
- `paymentType` (virtual_office / coworking_space / meeting_room)
- `spaceModel`, `space`, `spaceName`, `userEmail`, `userName`
- `totalAmount` (business total) and `amount` (gateway amount in paise in your dataset patterns)

### Invoice Keys
- `invoiceNumber`
- `user`, `partner`
- `subtotal`, `taxRate`, `total`, `status`

### Seat Booking Keys
- `paymentId` (string id referencing payment _id)
- `space`, `user`, `startTime/endTime`, `totalAmount`, `status`

### Space Keys
- properties: `city`, `area`, `partner`, `kycStatus`, `isActive`, `status`
- meetingrooms: `property`, `finalPricePerHour`, `finalPricePerDay`, `capacity`, `amenities`, `approvalStatus`
- coworkingspaces: `property`, `finalPricePerMonth`, `capacity`, `floors.tables.seats`

---

## 4) Universal Admin Answer Format

### A) Single Record (Booking/Payment/Invoice)
Return in this structure:
1. **Header**: entity + id
2. **Status & timeline** (createdAt/updatedAt where relevant)
3. **Financial summary** (amount/total/subtotal)
4. **Links** (userId, partnerId, spaceId/propertyId)
5. **Actionables** (KYC pending, missing invoice, anomaly)

### B) Analytics Report
Return in this structure:
1. **Totals**
2. **Breakdown** (by type/city/status)
3. **Top N** (partners/spaces/users)
4. **Anomalies**
5. **Next actions**

---

## 5) Clarification Rules (Ask minimal questions)
If admin says vague things like "revenue", "bookings", "users":
Ask only:
- **Timeframe**: Today / This week / This month / Custom range
Optional:
- City filter?
- Space type filter?

---

## 6) Join & Relationship Rules (Critical)

### 6.1 Booking <-> Payment linking
Preferred order:
1) If both contain `bookingNumber`, join by `bookingNumber` (best practice).
2) Seat bookings: join by `seatbookings.paymentId` -> `payments._id` (string -> ObjectId).
3) Fallback join (when no direct key): match by `(user + space + type/model + time window +/- 30 min)`.

### 6.2 Payment <-> Invoice linking
Preferred:
- If you store invoice `paymentId` (future), join by it.
Current dataset pattern:
- join by `(user + partner + description/type + createdAt window)`.

### 6.3 Booking <-> Space name
Use:
- `bookings.spaceSnapshot.name` (fast and reliable for display)

### 6.4 Space configuration linking
- `meetingrooms.property` -> `properties._id`
- `coworkingspaces.property` -> `properties._id`

### 6.5 Reviews linking
- `reviews.spaceModel` tells which collection the `space` belongs to.
- `reviews.space` -> resolve into:
  - CoworkingSpace / VirtualOffice / MeetingRoom (based on spaceModel)

---

## 7) Query Plans (Templates)

### 7.1 Find Booking by bookingNumber
Input: `FS-BKG-XXXX`
Plan:
- bookings.findOne({ bookingNumber })
Output:
- type, status, kycStatus, startDate, endDate, user, partner, spaceId, spaceSnapshot.name

### 7.2 List Bookings (timeframe)
Input: Today/Week/Month + optional filters (type/city)
Plan:
- bookings.find({ createdAt: range, ...(type filter) })
If city filter exists:
- join spaceId -> resolve city (if space collection has city) OR fallback to property mapping where available.

### 7.3 Find Payment by Razorpay Order ID
Input: `order_...`
Plan:
- payments.findOne({ razorpayOrderId })
Output:
- status, totalAmount, currency, paymentType, spaceModel, userEmail, userName, spaceName

### 7.4 Find Payment by Razorpay Payment ID
Input: `pay_...`
Plan:
- payments.findOne({ razorpayPaymentId })

### 7.5 Find Invoice by invoiceNumber
Input: `INV-...`
Plan:
- invoices.findOne({ invoiceNumber })
Output:
- subtotal, taxRate, total, status, user, partner

### 7.6 Seat Booking Details by seatbooking _id
Input: seatbooking id
Plan:
- seatbookings.findOne({ _id })
- if paymentId present: payments.findOne({ _id: ObjectId(paymentId) })

### 7.7 Meeting Rooms by capacity / amenities / price
Inputs: capacityMin, amenity list, priceMax
Plan:
- meetingrooms.find({
  capacity: { $gte: capacityMin },
  amenities: { $all: amenities },
  finalPricePerHour: { $lte: priceMax }
})
- join property -> properties to show city/area/name

### 7.8 Coworking Seats inventory summary
Input: propertyId
Plan:
- coworkingspaces.findOne({ property: propertyId })
Compute:
- total seats = sum of floors.tables.seats length
- active seats = count where seats.isActive=true

### 7.9 Reviews summary (avg rating)
Input: spaceModel or spaceId
Plan:
- reviews.aggregate([
  { $match: { spaceModel: "CoworkingSpace" } },
  { $group: { _id: "$space", avgRating: { $avg: "$rating" }, total: { $sum: 1 } } },
  { $sort: { avgRating: -1 } }
])

---

## 8) Revenue & Tax Logic

### 8.1 Prefer Invoices for Revenue Reporting
Use invoices.total for revenue KPI.
Fallback to payments.totalAmount if invoices missing.

### 8.2 GST
Your invoices store `taxRate: 18` and totals already computed.
Rule:
- If `taxAmount` is missing/0, treat total as authoritative.

---

## 9) Admin Analytics Recipes

### 9.1 Revenue by timeframe
- invoices.aggregate([
  { $match: { createdAt: range, status: "paid" } },
  { $group: { _id: null, revenue: { $sum: "$total" }, count: { $sum: 1 } } }
])

### 9.2 Revenue by partner
- invoices.aggregate([
  { $match: { createdAt: range, status: "paid" } },
  { $group: { _id: "$partner", revenue: { $sum: "$total" }, invoices: { $sum: 1 } } },
  { $sort: { revenue: -1 } }
])

### 9.3 Bookings by type
- bookings.aggregate([
  { $match: { createdAt: range } },
  { $group: { _id: "$type", bookings: { $sum: 1 } } }
])

### 9.4 Pending KYC (Bookings)
- bookings.find({ kycStatus: { $ne: "verified" }, status: "active" })

### 9.5 Pending KYC (Properties)
- properties.find({ kycStatus: { $ne: "verified" }, isDeleted: false })

### 9.6 Partner leads pending
- partnerinquiries.find({ status: "pending", isDeleted: false })

---

## 10) Anomaly / Audit Checks (Admin Alerts)

### 10.1 Orphan payments
Condition:
- payments.status = completed but no matching booking/invoice found (by bookingNumber or fallback join)

### 10.2 Booking active but invoice missing
Condition:
- bookings.status = active but no invoice for same user+partner+type within time window

### 10.3 Seat booking confirmed but payment not completed
Condition:
- seatbookings.status indicates confirmed AND linked payment.status != completed

### 10.4 Price mismatch
Compare:
- bookings.plan.price vs invoices.subtotal vs payments.totalAmount
Flag when absolute difference > tolerance.

---

## 11) Output Safety Rules
Even for Admin, never output:
- passwords
- auth tokens
- payment card details
- signatures

Mask any sensitive identifiers if needed in UI logs.

---

## 12) "If Data Missing" Fallback Response
If a query cannot be answered from the database:
Return:
- what was attempted
- which filters were used
- what extra input is needed (date range, id, etc.)

---

# End of flashspace_admin_query_logic.md
