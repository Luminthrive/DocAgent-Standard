import asyncio

from langchain_community.document_loaders import TextLoader, PyPDFLoader, BSHTMLLoader, Docx2txtLoader
from loguru import  logger

class ParseService:
    async def parse_document(self,file_path:str,file_type:str):
        logger.info(f"解析文档: {file_path} type={file_type} ")
        try:
            if file_type==".pdf":
                text=await asyncio.to_thread(self.load_pdf,file_path)
            elif file_type in [".txt",".md"]:
                text=await asyncio.to_thread(self.load_text,file_path)
            elif file_type ==".html":
                text=await asyncio.to_thread(self.load_html,file_path)
            elif file_type==".docx":
                text=await asyncio.to_thread(self.load_docx,file_path)
            elif file_type==".xlsx":
                text=await asyncio.to_thread(self.load_xlsx,file_path)
            elif file_type==".pptx":
                text=await asyncio.to_thread(self.load_pptx,file_path)
            else:
                raise ValueError(f"不受支持的文件类型: {file_type}")
        except ValueError:
            raise
        except Exception as e:
            logger.info(f"解析失败: {file_path} error={e}")
            return []
        return text



    @staticmethod
    def load_text(file_path:str):
        """txt/md 转化为text"""
        loader=TextLoader(file_path,encoding="utf-8")
        docs=loader.load()
        return docs

    @staticmethod
    def load_pdf(file_path:str):
        loader=PyPDFLoader(file_path)
        docs=loader.load()
        return docs

    @staticmethod
    def load_html(file_path:str):
        loader=BSHTMLLoader(
            file_path=file_path,
            open_encoding="utf-8",
            bs_kwargs={"features":"html.parser"}
        )
        docs=loader.load()
        return docs

    @staticmethod
    def load_docx(file_path:str):
        loader=Docx2txtLoader(file_path)
        docs=loader.load()
        return docs

    @staticmethod
    def load_xlsx(file_path:str):
        # loader=
        ...
    @staticmethod
    def load_pptx(file_path:str):
        ...