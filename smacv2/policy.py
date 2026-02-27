"""
SMACv2 Heuristic Policy for SHARPIE.

This module provides a default policy for SMACv2 environments with support
for participant guidance. The policy uses heuristics for unit control while
allowing participants to override or guide actions.
"""

import numpy as np
from typing import Dict, Optional, Any


class Policy:
    """
    Heuristic policy for SMACv2 with participant guidance support.

    The policy uses simple heuristics:
    - Attack nearest enemy if in range
    - Move towards nearest enemy if not in range
    - Accept participant guidance for movement direction or target selection

    Participant input interpretation:
    - Positive integers (1-4): Movement directions (N/S/E/W)
    - Higher integers: Attack specific enemy
    - -1 or None: No guidance, use heuristic
    """

    # Action IDs for SMACv2 (typical, may vary by map)
    NOOP = 0
    STOP = 1
    MOVE_NORTH = 2
    MOVE_SOUTH = 3
    MOVE_EAST = 4
    MOVE_WEST = 5

    # Guidance input mappings
    GUIDANCE_UP = 1
    GUIDANCE_DOWN = 2
    GUIDANCE_LEFT = 3
    GUIDANCE_RIGHT = 4
    GUIDANCE_NOOP = -1

    def __init__(
        self,
        room_name: str = "",
        env_wrapper: Optional[Any] = None
    ):
        """
        Initialize the policy.

        Args:
            room_name: Name of the experiment room (for logging/debugging)
            env_wrapper: Reference to the environment wrapper (for action masking)
        """
        self.room_name = room_name
        self.env_wrapper = env_wrapper

        # Movement action mapping from guidance
        self.guidance_to_move = {
            self.GUIDANCE_UP: self.MOVE_NORTH,
            self.GUIDANCE_DOWN: self.MOVE_SOUTH,
            self.GUIDANCE_RIGHT: self.MOVE_EAST,
            self.GUIDANCE_LEFT: self.MOVE_WEST,
        }

    # Attack action base index (actions 6+ are typically attack actions)
    ATTACK_BASE = 6

    def predict(
        self,
        observation: np.ndarray,
        participant_input: Optional[int] = None,
        action_mask: Optional[np.ndarray] = None
    ) -> int:
        """
        Predict an action based on observation and optional participant guidance.

        Args:
            observation: Agent's observation array from SMACv2
            participant_input: Optional guidance from participant
                - -1 or None: Use heuristic
                - 1-4: Movement directions
                - Higher: Attack specific enemy (if available)
            action_mask: Binary mask of available actions (1 = available)

        Returns:
            action: The action to take (integer)
        """
        # If participant provided movement guidance, check if valid
        if participant_input is not None and participant_input != self.GUIDANCE_NOOP:
            if participant_input in self.guidance_to_move:
                move_action = self.guidance_to_move[participant_input]
                if action_mask is None or (move_action < len(action_mask) and action_mask[move_action]):
                    return move_action

        # Use heuristic with action mask
        return self._heuristic_action(observation, action_mask)

    def _heuristic_action(self, observation: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        """
        Select action using simple heuristics with action mask validation.

        Strategy:
        1. Try to attack nearest enemy (actions 6+)
        2. Try STOP if available
        3. Fall back to first available action

        Args:
            observation: Agent's observation array
            action_mask: Binary mask of available actions (1 = available)

        Returns:
            action: Selected action (guaranteed to be valid)
        """
        if action_mask is not None:
            # Try attack actions first (typically actions 6+)
            attack_actions = np.flatnonzero(action_mask[self.ATTACK_BASE:])
            if len(attack_actions) > 0:
                return self.ATTACK_BASE + int(attack_actions[0])

            # Try STOP if available
            if self.STOP < len(action_mask) and action_mask[self.STOP]:
                return self.STOP

            # Fall back to first available action
            available = np.flatnonzero(action_mask)
            if len(available) > 0:
                return int(available[0])

            # No actions available (shouldn't happen, but safety)
            return self.NOOP

        # No action mask provided - return STOP as safe default
        return self.STOP

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_observation: np.ndarray
    ):
        """
        Update policy based on experience (no-op for heuristic policy).

        Args:
            observation: Previous observation
            action: Action taken
            reward: Reward received
            done: Whether episode ended
            next_observation: Next observation
        """
        # Heuristic policy doesn't learn
        pass

# Create default policy instance
policy = Policy()