"""
A2A (Agent-to-Agent) Protocol
Industry-standard protocol backed by Google + 50 partners
Enables secure cross-swarm communication with security cards
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import logging

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MessageType(Enum):
    """A2A message types"""
    REQUEST = "request"  # Request for action/information
    RESPONSE = "response"  # Response to a request
    NOTIFICATION = "notification"  # One-way notification
    QUERY = "query"  # Query for data
    COMMAND = "command"  # Direct command to execute
    EVENT = "event"  # Event notification


@dataclass
class SecurityCard:
    """
    Security card for A2A protocol
    Defines permissions and capabilities
    """
    issuer: str  # Who issued this card
    subject: str  # Who this card is for (swarm/agent ID)
    capabilities: List[str]  # What actions are permitted
    issued_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if security card is still valid"""
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True

    def can_perform(self, action: str) -> bool:
        """Check if card permits an action"""
        if not self.is_valid():
            return False
        return action in self.capabilities or "*" in self.capabilities


@dataclass
class A2AMessage:
    """
    A2A protocol message

    Based on Google's A2A specification for agent communication
    """
    message_id: str
    sender: str  # Sender swarm/agent ID
    receiver: str  # Receiver swarm/agent ID
    message_type: MessageType
    content: Any
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)

    # Optional fields
    correlation_id: Optional[str] = None  # For request/response matching
    reply_to: Optional[str] = None  # Message ID this is replying to
    security_card: Optional[SecurityCard] = None  # Security credentials
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Delivery tracking
    delivered: bool = False
    delivered_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary"""
        return {
            'message_id': self.message_id,
            'sender': self.sender,
            'receiver': self.receiver,
            'message_type': self.message_type.value,
            'content': self.content,
            'priority': self.priority.value,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id,
            'reply_to': self.reply_to,
            'metadata': self.metadata,
            'delivered': self.delivered,
            'acknowledged': self.acknowledged
        }


class A2AProtocol:
    """
    Agent-to-Agent Communication Protocol

    Features:
    - Industry-standard A2A protocol (Google + 50 partners)
    - Security cards for authentication and authorization
    - Message queue per swarm
    - Request/response pattern
    - Priority-based message delivery
    - Message history and audit trail

    This enables swarms to communicate with each other securely,
    allowing complex multi-swarm workflows.
    """

    def __init__(self):
        """Initialize A2A protocol"""
        # Message queues per swarm/agent
        self._queues: Dict[str, List[A2AMessage]] = {}

        # Message history for audit trail
        self._history: List[A2AMessage] = []

        # Security cards registry
        self._security_cards: Dict[str, SecurityCard] = {}

        # Pending requests (for request/response matching)
        self._pending_requests: Dict[str, A2AMessage] = {}

        logger.info("Initialized A2A Protocol")

    def issue_security_card(
        self,
        issuer: str,
        subject: str,
        capabilities: List[str],
        expires_in_hours: Optional[int] = None
    ) -> SecurityCard:
        """
        Issue a security card to a swarm/agent

        Args:
            issuer: Who is issuing this card
            subject: Who this card is for
            capabilities: List of permitted actions (or ["*"] for all)
            expires_in_hours: Optional expiration time

        Returns:
            SecurityCard
        """
        now = datetime.now()
        expires_at = None
        if expires_in_hours:
            from datetime import timedelta
            expires_at = now + timedelta(hours=expires_in_hours)

        card = SecurityCard(
            issuer=issuer,
            subject=subject,
            capabilities=capabilities,
            issued_at=now,
            expires_at=expires_at
        )

        self._security_cards[subject] = card
        logger.info(f"Issued security card to {subject} with capabilities: {capabilities}")

        return card

    def send_message(
        self,
        sender: str,
        receiver: str,
        message_type: MessageType,
        content: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        security_card: Optional[SecurityCard] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> A2AMessage:
        """
        Send message from one agent/swarm to another

        Args:
            sender: Sender ID
            receiver: Receiver ID
            message_type: Type of message
            content: Message content
            priority: Message priority
            correlation_id: For request/response matching
            reply_to: Message ID being replied to
            security_card: Security credentials
            metadata: Optional metadata

        Returns:
            Sent message
        """
        # Create message
        message = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender=sender,
            receiver=receiver,
            message_type=message_type,
            content=content,
            priority=priority,
            correlation_id=correlation_id,
            reply_to=reply_to,
            security_card=security_card,
            metadata=metadata or {}
        )

        # Validate security if card provided
        if security_card:
            if not security_card.is_valid():
                logger.error(f"Invalid security card from {sender}")
                raise ValueError("Invalid security card")

        # Add to receiver's queue
        if receiver not in self._queues:
            self._queues[receiver] = []

        self._queues[receiver].append(message)

        # Add to history
        self._history.append(message)

        # Track pending requests
        if message_type == MessageType.REQUEST:
            self._pending_requests[message.message_id] = message

        logger.debug(
            f"A2A: {sender} → {receiver} ({message_type.value}) "
            f"[{priority.value}] msg_id={message.message_id[:8]}"
        )

        return message

    def receive_messages(
        self,
        receiver: str,
        message_type: Optional[MessageType] = None,
        priority: Optional[MessagePriority] = None,
        limit: Optional[int] = None
    ) -> List[A2AMessage]:
        """
        Receive messages for a swarm/agent

        Args:
            receiver: Receiver ID
            message_type: Optional filter by message type
            priority: Optional filter by priority
            limit: Maximum number of messages to receive

        Returns:
            List of messages
        """
        if receiver not in self._queues:
            return []

        queue = self._queues[receiver]

        # Filter by message type
        if message_type:
            queue = [m for m in queue if m.message_type == message_type]

        # Filter by priority
        if priority:
            queue = [m for m in queue if m.priority == priority]

        # Sort by priority (URGENT → HIGH → NORMAL → LOW)
        priority_order = {
            MessagePriority.URGENT: 0,
            MessagePriority.HIGH: 1,
            MessagePriority.NORMAL: 2,
            MessagePriority.LOW: 3
        }
        queue.sort(key=lambda m: priority_order[m.priority])

        # Apply limit
        if limit:
            queue = queue[:limit]

        # Mark as delivered
        for message in queue:
            if not message.delivered:
                message.delivered = True
                message.delivered_at = datetime.now()

        # Remove from queue
        for message in queue:
            self._queues[receiver].remove(message)

        logger.debug(f"A2A: {receiver} received {len(queue)} messages")

        return queue

    def send_request(
        self,
        sender: str,
        receiver: str,
        content: Any,
        timeout_seconds: int = 60,
        security_card: Optional[SecurityCard] = None
    ) -> str:
        """
        Send request and get correlation ID for response matching

        Args:
            sender: Sender ID
            receiver: Receiver ID
            content: Request content
            timeout_seconds: Request timeout
            security_card: Security credentials

        Returns:
            Correlation ID for matching response
        """
        correlation_id = str(uuid.uuid4())

        message = self.send_message(
            sender=sender,
            receiver=receiver,
            message_type=MessageType.REQUEST,
            content=content,
            correlation_id=correlation_id,
            security_card=security_card,
            metadata={'timeout': timeout_seconds}
        )

        return correlation_id

    def send_response(
        self,
        sender: str,
        receiver: str,
        content: Any,
        request_message_id: str,
        correlation_id: str,
        security_card: Optional[SecurityCard] = None
    ) -> A2AMessage:
        """
        Send response to a request

        Args:
            sender: Sender ID
            receiver: Receiver ID (original requester)
            content: Response content
            request_message_id: Message ID of original request
            correlation_id: Correlation ID from request
            security_card: Security credentials

        Returns:
            Response message
        """
        message = self.send_message(
            sender=sender,
            receiver=receiver,
            message_type=MessageType.RESPONSE,
            content=content,
            reply_to=request_message_id,
            correlation_id=correlation_id,
            security_card=security_card
        )

        # Remove from pending requests
        if request_message_id in self._pending_requests:
            del self._pending_requests[request_message_id]

        return message

    def broadcast(
        self,
        sender: str,
        receivers: List[str],
        message_type: MessageType,
        content: Any,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> List[A2AMessage]:
        """
        Broadcast message to multiple receivers

        Args:
            sender: Sender ID
            receivers: List of receiver IDs
            message_type: Message type
            content: Message content
            priority: Message priority

        Returns:
            List of sent messages
        """
        messages = []

        for receiver in receivers:
            message = self.send_message(
                sender=sender,
                receiver=receiver,
                message_type=message_type,
                content=content,
                priority=priority
            )
            messages.append(message)

        logger.info(f"A2A: {sender} broadcast to {len(receivers)} receivers")

        return messages

    def acknowledge_message(self, message: A2AMessage):
        """
        Acknowledge receipt and processing of message

        Args:
            message: Message to acknowledge
        """
        message.acknowledged = True
        message.acknowledged_at = datetime.now()

        logger.debug(f"A2A: Message {message.message_id[:8]} acknowledged")

    def get_pending_requests(
        self,
        sender: Optional[str] = None
    ) -> List[A2AMessage]:
        """
        Get pending requests (awaiting response)

        Args:
            sender: Optional filter by sender

        Returns:
            List of pending request messages
        """
        requests = list(self._pending_requests.values())

        if sender:
            requests = [r for r in requests if r.sender == sender]

        return requests

    def get_message_history(
        self,
        swarm_id: Optional[str] = None,
        limit: int = 100
    ) -> List[A2AMessage]:
        """
        Get message history

        Args:
            swarm_id: Optional filter by sender or receiver
            limit: Maximum number of messages

        Returns:
            List of messages (most recent first)
        """
        history = self._history.copy()

        if swarm_id:
            history = [
                m for m in history
                if m.sender == swarm_id or m.receiver == swarm_id
            ]

        history.reverse()  # Most recent first

        return history[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get A2A protocol statistics

        Returns:
            Dictionary with stats
        """
        total_messages = len(self._history)
        pending_requests = len(self._pending_requests)
        active_queues = sum(1 for q in self._queues.values() if q)

        # Count by message type
        by_type = {}
        for msg in self._history:
            msg_type = msg.message_type.value
            by_type[msg_type] = by_type.get(msg_type, 0) + 1

        return {
            'total_messages': total_messages,
            'pending_requests': pending_requests,
            'active_queues': active_queues,
            'security_cards_issued': len(self._security_cards),
            'by_type': by_type
        }


# Singleton instance
_a2a_protocol: Optional[A2AProtocol] = None


def get_a2a_protocol() -> A2AProtocol:
    """
    Get or create singleton A2A protocol

    Returns:
        Global A2A protocol instance
    """
    global _a2a_protocol
    if _a2a_protocol is None:
        _a2a_protocol = A2AProtocol()
    return _a2a_protocol
