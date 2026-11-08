"""Knowledge base for the Ola Business-Operations / Customer-Support agent.

12 documents (4-5 sentences each), written for this project. The policies are
FICTIONAL and illustrative - they are not Ola's real internal policies.
"""

KB_DOCS = [
    {
        "doc_id": "kb-01",
        "topic": "ticket-priority classification rules",
        "text": (
            "Every incoming ticket is given one of four priority levels when it is created. "
            "P1 (critical) covers rider safety incidents, a rider being charged twice for one trip, and total loss of account access during an active ride. "
            "P2 (high) covers failed refunds, repeated app crashes, and drivers who cannot go online to accept trips. "
            "P3 (normal) covers fare disputes under 500 rupees and product defects, while P4 (low) covers feature requests and general inquiries. "
            "The first agent to read a ticket may raise its priority, but only a team lead can lower it."
        ),
    },
    {
        "doc_id": "kb-02",
        "topic": "SLA-by-severity policy",
        "text": (
            "Response and resolution targets depend on ticket severity. "
            "A P1 ticket gets a first response within 15 minutes and a resolution target of 4 hours, around the clock. "
            "A P2 ticket gets a first response within 1 hour and a resolution target of 12 hours. "
            "A P3 ticket gets a first response within 4 hours and a resolution target of 48 hours, and a P4 ticket gets a response within 24 hours and a resolution target of 5 business days. "
            "Any ticket that misses its resolution target is marked as an SLA breach and appears on the team lead dashboard."
        ),
    },
    {
        "doc_id": "kb-03",
        "topic": "escalation matrix",
        "text": (
            "Tickets move up through three tiers of support. "
            "A Tier 1 agent escalates to a Tier 2 specialist when a ticket is still open after 80 percent of its SLA window has passed, or when the customer asks to speak to a supervisor. "
            "Tier 2 escalates to the duty manager for any P1 ticket, any suspected fraud, or any compensation request above 2,000 rupees. "
            "The regional operations head is involved only when an incident affects more than 500 customers or carries legal risk. "
            "Every escalation must include a one-line summary and a list of the steps already tried."
        ),
    },
    {
        "doc_id": "kb-04",
        "topic": "refund and compensation policy",
        "text": (
            "A refund for a wrongly charged fare goes back to the original payment method within 5 to 7 business days after approval. "
            "Refunds to the in-app wallet are instant. "
            "An agent may approve a refund of up to 1,000 rupees alone, and anything higher needs Tier 2 approval. "
            "Goodwill compensation for a poor experience is capped at 300 rupees per ticket and is paid as ride credit, never as cash."
        ),
    },
    {
        "doc_id": "kb-05",
        "topic": "customer-communication-channel policy",
        "text": (
            "Support talks to customers through in-app chat, email, and phone, and each channel has a specific role. "
            "In-app chat is the default channel for every ticket and is answered inside the SLA window. "
            "Phone callbacks are offered only for P1 and P2 tickets, or when a customer cannot use the app. "
            "Email is reserved for formal outcomes such as refund confirmations and account-closure notices. "
            "Agents never ask for a full card number or a one-time password on any channel, and they never move a conversation to a personal messaging app."
        ),
    },
    {
        "doc_id": "kb-06",
        "topic": "business-hours and holiday support policy",
        "text": (
            "Live chat and phone support run from 6 AM to 11 PM IST every day of the year, including national holidays. "
            "Between 11 PM and 6 AM only P1 safety and payment tickets are handled by the night desk, and all other tickets wait in the queue until morning. "
            "On public holidays the SLA clock is paused for P3 and P4 tickets, while the clocks for P1 and P2 tickets keep running. "
            "Current support hours are shown in the Help section of the app."
        ),
    },
    {
        "doc_id": "kb-07",
        "topic": "repeat-complaint handling policy",
        "text": (
            "A repeat complaint is a new ticket about the same issue raised within 14 days of an earlier ticket on that topic. "
            "Repeat tickets are linked to the original instead of being treated as new, and their priority is raised by one level automatically. "
            "The agent must read the earlier resolution notes before replying, and the customer should never be asked to explain the problem again. "
            "A third complaint about the same issue goes straight to Tier 2 for a root-cause review."
        ),
    },
    {
        "doc_id": "kb-08",
        "topic": "service-credit policy",
        "text": (
            "Service credits are ride credits added to a customer wallet when we miss a commitment. "
            "A ticket that breaches its SLA earns a credit of 50 rupees for P3, 100 rupees for P2, and 200 rupees for P1. "
            "Credits expire 90 days after they are issued and cannot be withdrawn as cash. "
            "A customer can receive at most three service credits in one calendar month."
        ),
    },
    {
        "doc_id": "kb-09",
        "topic": "feedback-collection process",
        "text": (
            "A short feedback survey is sent automatically as soon as a ticket moves to Resolved. "
            "The survey asks for a satisfaction rating from 1 to 5 and one optional comment. "
            "It stays open for 3 days, and a single reminder is sent after 24 hours. "
            "Any rating of 2 or below reopens the ticket for a quality review by a team lead within one business day. "
            "Survey scores are never used to judge an individual agent on the basis of one response."
        ),
    },
    {
        "doc_id": "kb-10",
        "topic": "VIP-customer handling policy",
        "text": (
            "Customers on the Prime tier, or flagged as VIP by the business team, are served from a dedicated priority queue. "
            "Their tickets start one priority level higher than normal, and the first-response target is cut to 10 minutes. "
            "A named senior agent owns the whole ticket from start to finish so the customer deals with one person. "
            "VIP status never changes the refund limits or the security checks that apply to everyone."
        ),
    },
    {
        "doc_id": "kb-11",
        "topic": "outage-communication protocol",
        "text": (
            "When a platform outage affects bookings or payments, the incident manager declares an outage within 15 minutes of confirming it. "
            "A banner appears in the app and a status message is posted within 30 minutes, and updates follow every 60 minutes until the issue is fixed. "
            "Support agents use an approved script and never guess at causes or fix times. "
            "After recovery, affected customers receive a short apology and any applicable service credit automatically."
        ),
    },
    {
        "doc_id": "kb-12",
        "topic": "data-retention policy for tickets",
        "text": (
            "Resolved ticket records are kept in the support system for 24 months and are then permanently deleted. "
            "Chat transcripts and call recordings linked to a ticket are removed after 12 months unless a legal hold applies. "
            "Phone numbers and email addresses inside old tickets are masked after 6 months. "
            "A customer may ask for their ticket history to be deleted, and the request must be completed within 30 days unless the data is needed for a fraud or legal case."
        ),
    },
]

KB_BY_ID = {d["doc_id"]: d for d in KB_DOCS}

if __name__ == "__main__":
    import re
    for d in KB_DOCS:
        n = len(re.findall(r"[.!?](?:\s|$)", d["text"]))
        print(d["doc_id"], f"{n} sentences", "-", d["topic"])
        assert 2 <= n <= 5
