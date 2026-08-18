

class AgentService:
    ...

    async def handle_human_confirmation(self, session_id, user_id, confirmation):
        pass

    def run_stream(self, session_id, user_id, message):
        pass

    async def run(self, session_id, user_id, message, enable_trace):
        pass