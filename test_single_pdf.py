import os
import csv
import pandas as pd
import umap
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import NarrativeText, Title
from src.embeddings.e5_embedding import MultilingualE5LargeInstruct
from src.embeddings.jina_embedding import JinaEmbeddingsV3
from src.embeddings.bge_embedding import BgeM3Embedding
from src.embeddings.qwen_embedding import Qwen3Embedding06B


# --- 설정: 여기에서 테스트할 PDF 파일과 사용할 모델을 지정하세요 ---
# PDF 파일 이름 (프로젝트 루트 디렉토리에 파일이 있어야 합니다)
# PDF_TO_TEST = "iPhone_17_Pro_and_iPhone_17_Pro_Max_PER_Sept2025.pdf"  # 예시: "mydocument.pdf"
PDF_TO_TEST = "SM-S93X_S92X_UG_16_Kor_Rev.1.0_250912.pdf"

# 사용할 임베딩 모델 클래스와 결과 파일에 사용할 이름
EMBEDDING_MODEL_CLASS = MultilingualE5LargeInstruct
MODEL_NAME_FOR_FILENAME = "multilingual_e5_large"
# EMBEDDING_MODEL_CLASS = JinaEmbeddingsV3
# MODEL_NAME_FOR_FILENAME = "jina_v3"
# EMBEDDING_MODEL_CLASS = BgeM3Embedding
# MODEL_NAME_FOR_FILENAME = "bge_m3"
# EMBEDDING_MODEL_CLASS = Qwen3Embedding06B
# MODEL_NAME_FOR_FILENAME = "qwen3_06b"


def process_single_pdf_for_atlas():
    """
    설정된 단일 PDF 파일을 파싱하고, 텍스트를 청크로 나누어 임베딩을 생성한 후,
    Embedding Atlas 시각화를 위한 CSV 파일로 저장합니다.
    """
    # 1. 대상 PDF 파일 존재 여부 확인
    if not os.path.exists(PDF_TO_TEST):
        print(f"오류: '{PDF_TO_TEST}' 파일을 찾을 수 없습니다.")
        print("파일이 프로젝트 루트 디렉토리에 있는지, 파일 이름이 정확한지 확인해주세요.")
        return

    # 2. 임베딩 모델 초기화
    print(f"'{MODEL_NAME_FOR_FILENAME}' 모델을 로드합니다...")
    try:
        embedder = EMBEDDING_MODEL_CLASS()
        # 모델의 임베딩 차원 확인 (multilingual-e5-large-instruct는 1024차원)
        embedding_dim = 1024
        print("모델 로드 완료.")
    except Exception as e:
        print(f"모델 로드 중 오류가 발생했습니다: {e}")
        return

    # 3. PDF 파싱 및 텍스트 청크 추출
    print(f"'{PDF_TO_TEST}' 파일을 unstructured로 파싱합니다...")
    try:
        elements = partition_pdf(filename=PDF_TO_TEST, strategy="auto", languages=['kor', 'eng'])
        
        # 의미 있는 텍스트를 가진 요소만 필터링 (예: NarrativeText, Title)
        text_chunks = [
            el.text for el in elements 
            if isinstance(el, (NarrativeText, Title)) and len(el.text) > 20
        ]

        if not text_chunks:
            print(f"경고: '{PDF_TO_TEST}'에서 유의미한 텍스트 청크를 추출할 수 없습니다.")
            return
    except Exception as e:
        print(f"PDF 파싱 중 오류가 발생했습니다: {e}")
        return

    # 4. 임베딩 생성 (배치 처리)
    print(f"{len(text_chunks)}개의 텍스트 청크에 대한 임베딩을 생성합니다...")
    embeddings = embedder.embed_documents(text_chunks)

    # 5. UMAP으로 2D 프로젝션 생성
    print("UMAP을 사용하여 2D 프로젝션을 생성합니다...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    projection = reducer.fit_transform(embeddings)

    # 6. Pandas DataFrame 생성 및 Parquet 파일로 저장
    print("결과를 Parquet 파일로 저장합니다...")
    df = pd.DataFrame({
        'id': range(len(text_chunks)),
        'text': text_chunks
    })
    df['projection_x'] = projection[:, 0]
    df['projection_y'] = projection[:, 1]
    # 원본 고차원 임베딩 벡터를 DataFrame에 추가
    df['embedding'] = list(embeddings)

    base_filename = os.path.splitext(PDF_TO_TEST)[0]
    output_filename = f"{base_filename}_{MODEL_NAME_FOR_FILENAME}_atlas_projection.parquet"
    
    df.to_parquet(output_filename, index=False)

    print(f"\n작업이 완료되었습니다. '{output_filename}' 파일을 확인하세요.")
    print("\n다음 명령어를 사용하여 Embedding Atlas를 실행하세요:")
    print(f"embedding-atlas {output_filename} --x projection_x --y projection_y --text text")


if __name__ == '__main__':
    process_single_pdf_for_atlas() 