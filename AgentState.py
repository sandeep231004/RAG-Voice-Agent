class AgentState:
    """Class representing the agent's internal state."""

    def __init__(self):
        self.active = False  # Whether the agent is currently active
        self.current_task = None  # Current task the agent is handling
        self.goals = []  # List of goals assigned to the agent
        self.waiting_for_user_input = False  # If agent is waiting for user response
        self.last_tool_result = None  # Output from last tool used
        self.tools_used_in_current_task = []  # Tools used for the current task
        self.user_preferences = {}  # Any preferences provided by the user

    def set_active(self, active: bool):
        """Set the agent's active status."""
        self.active = active

    def set_current_task(self, task: str):
        """Set the current task and reset tool usage for it."""
        self.current_task = task
        self.tools_used_in_current_task = []

    def add_goal(self, goal: str):
        """Add a goal for the agent."""
        self.goals.append(goal)

    def mark_waiting_for_input(self, waiting: bool):
        """Mark whether the agent is waiting for user input."""
        self.waiting_for_user_input = waiting

    def record_tool_use(self, tool_name: str):
        """Record a tool used in the current task."""
        self.tools_used_in_current_task.append(tool_name)

    def set_user_preference(self, key: str, value):
        """Store a user preference."""
        self.user_preferences[key] = value

    def get_status_summary(self) -> dict:
        """Return the agent's current state as a dictionary."""
        return {
            "active": self.active,
            "current_task": self.current_task,
            "goals": self.goals,
            "waiting_for_input": self.waiting_for_user_input,
            "user_preferences": self.user_preferences
        }
