

class SessionService:
    ...

    async def create_session(self, user_id, kb_id, title):
        pass

    async def list_sessions(self, user_id, limit, offset):
        pass

    async def get_session(self, session_id, user_id):
        pass

    async def list_messages(self, session_id):
        pass

    async def delete_session(self, session_id):
        pass