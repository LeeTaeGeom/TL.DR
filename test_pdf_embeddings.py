import os
import csv
from unstructured.partition.pdf import partition_pdf
from src.embeddings.multilingual_e5_large_instruct import MultilingualE5LargeInstruct
from src.embeddings.jina_embeddings_v3 import JinaEmbeddingsV3


def create_pdf_embeddings_per_model():
    """
    unstructured 라이브러리를 사용하여 PDF 파일을 파싱하고,
    구현된 각 임베딩 모델을 사용하여 임베딩을 생성합니다.
    결과는 '원본파일이름_모델이름_embedding.csv' 형식의 개별 파일로 저장됩니다.
    """
    # 1. 임베딩 모델들 초기화
    print("임베딩 모델들을 로드합니다...")
    try:
        models = {
            "multilingual_e5_large": MultilingualE5LargeInstruct(),
            "jina_v3": JinaEmbeddingsV3()
        }
        print(f"{len(models)}개의 모델 로드 완료: {list(models.keys())}")
    except Exception as e:
        print(f"모델 로드 중 오류가 발생했습니다: {e}")
        print("'trust_remote_code=True'가 필요한 모델일 수 있습니다.")
        print("또는 'uv pip install -r requirements.txt'를 실행하여 의존성을 확인하세요.")
        return

    # 2. 현재 디렉토리에서 PDF 파일 찾기
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("현재 디렉토리에서 PDF 파일을 찾을 수 없습니다.")
        return
    print(f"총 {len(pdf_files)}개의 PDF 파일을 찾았습니다: {pdf_files}")

    total_files_generated = 0

    # 3. 각 PDF 파일과 각 모델에 대해 처리
    for pdf_file in pdf_files:
        print(f"\n--- '{pdf_file}' 파일 처리 시작 ---")
        try:
            # PDF 파싱은 모델마다 반복할 필요 없으므로 한 번만 수행
            print(f"  - (1/3) unstructured로 텍스트를 추출합니다...")
            elements = partition_pdf(filename=pdf_file, strategy="auto")
            text_content = "\n\n".join([el.text for el in elements])

            if not text_content.strip():
                print(f"  - 경고: '{pdf_file}'에서 텍스트를 추출할 수 없습니다. 건너뜁니다.")
                continue

            # 각 모델에 대해 임베딩 및 저장
            for model_name, embedder in models.items():
                print(f"  - (2/3) '{model_name}' 모델로 임베딩을 생성합니다...")
                embedding = embedder.embed_query(text_content)
                embedding_str = ",".join(map(str, embedding))

                # 동적 파일 이름 생성
                base_filename = os.path.splitext(pdf_file)[0]
                output_filename = f"{base_filename}_{model_name}_embedding.csv"

                print(f"  - (3/3) 결과를 '{output_filename}' 파일에 저장합니다...")
                with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['original_filename', 'embedding_model', 'extracted_text', 'embedding'])
                    writer.writerow([pdf_file, model_name, text_content, embedding_str])

                total_files_generated += 1

        except Exception as e:
            print(f"  - 오류: '{pdf_file}' 처리 중 오류 발생: {e}")

    print(f"\n모든 작업이 완료되었습니다. 총 {total_files_generated}개의 CSV 파일을 생성했습니다.")


if __name__ == '__main__':
    create_pdf_embeddings_per_model()
