
from fastapi import APIRouter, Depends, Query, HTTPException, UploadFile, File

from core.auth import get_current_user_id
from core.dependencies import get_knowledge_service, get_document_service, get_vector_service
from schema.knowledge_schema import CreateKnowledgeBaseRequest, KnowledgeBaseResponse, UploadResponse, DocumentResponse, \
    RetrieveRequest, RetrieveResponse


router=APIRouter()
#接下里将这三个服务改为依赖注入
# kb_service=KnowledgeService()
# doc_service=DocumentService()
# vector_service=VectorService()

@router.post("/bases",response_model=KnowledgeBaseResponse,summary="创建知识库")
async def create_knowledge_base(
        request:CreateKnowledgeBaseRequest,
        user_id:str=Depends(get_current_user_id),
        kb_service=Depends(get_knowledge_service)
):
    """
    创建知识库
    :param request:
    :param user_id:
    :return:
    """
    kb=await kb_service.create_knowledge_base(
        user_id=user_id,
        name=request.name,
        description=request.description
    )
    return kb

@router.get("/bases",response_model=list[KnowledgeBaseResponse],summary="获取知识库列表")
async def list_knowledge_bases(
        limit:int=Query(50,ge=1,le=100,description="每页数量"),
        offset:int=Query(0,ge=0,description="偏移量"),
        user_id:str=Depends(get_current_user_id),
        kb_service=Depends(get_knowledge_service)
):
    """
    列出知识库
    :param limit:
    :param offset:
    :param user_id:
    :return:
    """
    kbs=await kb_service.list_knowledge_bases(
        user_id=user_id,
        limit=limit,
        offset=offset
    )
    return kbs

@router.delete("/bases/{kb_id}",summary="删除知识库")
async def delete_knowledge_base(
        kb_id:str,
        user_id:str=Depends(get_current_user_id),
        kb_service=Depends(get_knowledge_service)
):
    """
    删除知识库及其所有文档，分块向量
    :param kb_id:
    :param user_id:
    :return:
    """
    kb=await kb_service.get_knowledge_base(kb_id,user_id)
    if not kb:
        raise HTTPException(status_code=404,detail="知识库不存在")
    await kb_service.delete_knowledge_base(kb_id)
    return {"status":"deleted","kb_id":kb_id}

@router.post("/bases/{kb_id}/documents",response_model=UploadResponse,summary="上传文档")
async def upload_document(
        kb_id:str,
        file:UploadFile=File(...,description="文档文件"),
        user_id:str=Depends(get_current_user_id),
        kb_service=Depends(get_knowledge_service),
        doc_service=Depends(get_document_service)
):
    """
    上传文档到知识库
    :param kb_id:
    :param file:
    :param user_id:
    :return:
    """
    kb=await kb_service.get_knowledge_base(kb_id,user_id=user_id)
    if not kb:
        raise HTTPException(status_code=404,detail="知识库不存在")
    allowed_extensions=[".pdf",".docx",".xlsx",".pptx",".txt",".md"]
    file_ext="."+file.filename.rsplit(".",1)[-1].lower() if '.' in file.filename else ""
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=404,
            detail=f"不支持的文件格式:{file_ext},支持{",".join(allowed_extensions)}"
        )
    result=await doc_service.upload_document(
        kb_id,
        file=file,
        user_id=user_id
    )
    return result

@router.get("/bases/{kb_id}/documents",response_model=list[DocumentResponse],summary="文档列表")
async def list_documents(
        kb_id:str,
        user_id:str=Depends(get_current_user_id),
        kb_service=Depends(get_knowledge_service),
        doc_service=Depends(get_document_service)
):
    """获取知识库下所有文档"""
    kb=await kb_service.get_knowledge_base(kb_id,user_id)
    if not kb:
        raise HTTPException(status_code=404,detail="知识库不存在")
    docs=await doc_service.list_documents(kb_id)
    return docs

@router.delete("/bases/{kb_id}/documents/{doc_id}",summary="删除文档")
async def delete_document(
        kb_id:str,
        doc_id:str,
        user_id:str=Depends(get_current_user_id),
        kb_service=Depends(get_knowledge_service),
        doc_service=Depends(get_document_service),
):
    """
    删除特定文档相关向量
    :param kb_id:
    :param doc_id:
    :param user_id:
    :return:
    """
    kb=await kb_service.get_knowledge_base(kb_id,user_id)
    if not kb:
        raise HTTPException(status_code=404,detail="知识库不存在")
    doc=await doc_service.get_document(doc_id,kb_id)
    if not doc:
        raise HTTPException(status_code=404,detail="文档不存在")
    await doc_service.delete_document(kb_id,doc_id)
    return {"status":"deleted","doc_id":doc_id}

@router.post("/bases/{kb_id}/retrieve",response_model=RetrieveResponse,summary="独立检索接口")
async def retrieve(
        kb_id:str,
        request:RetrieveRequest,
        user_id:str=Depends(get_current_user_id),
        kb_service=Depends(get_knowledge_service),
        vector_service=Depends(get_vector_service)
):
    kb=await kb_service.get_knowledge_base(kb_id,user_id)
    if not kb:
        raise HTTPException(status_code=404,detail="知识库不存在")
    results=await vector_service.search(
        kb_id=kb_id,
        query=request.query,
        top_k=request.top_k
    )
    return RetrieveResponse(
        query=request.query,
        results=results,
        count=len(results)
    )


