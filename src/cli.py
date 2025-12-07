#!/usr/bin/env python3
"""TL.DR RAG CLI - Interactive question-answering system."""

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processors.processor_factory import get_pdf_processor  # noqa: E402
from src.embeddings.embedding_factory import (  # noqa: E402
    EmbeddingFactory,
    get_embedding,
)
from src.llms.ollama_llm import OllamaLLM  # noqa: E402
from src.llms.openai_llm import OpenAILLM  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402
from src.vector_dbs.chroma_db import ChromaVectorDB  # noqa: E402


console = Console()


def create_pipeline(args) -> RAGPipeline:
    """Create RAG pipeline from arguments."""
    # Document processor
    console.print("[dim]📄 문서 처리기 초기화 중...[/dim]")
    processor = get_pdf_processor(args.processor)

    # Embedding
    console.print(f"[dim]🔢 임베딩 모델 로딩 중 ({args.embedding})...[/dim]")
    embedding = get_embedding(args.embedding)

    # Vector DB
    console.print("[dim]🗄️  벡터 DB 초기화 중...[/dim]")
    persist_dir = args.db_path if args.persist else None
    vector_db = ChromaVectorDB(
        collection_name=args.collection,
        persist_directory=persist_dir,
    )

    # LLM
    console.print(f"[dim]🤖 LLM 초기화 중 ({args.llm})...[/dim]")
    if args.llm == "openai":
        if not os.environ.get("OPENAI_API_KEY") and not args.api_key:
            console.print(
                "[red]❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.[/red]"
            )
            sys.exit(1)
        llm = OpenAILLM(model=args.model, api_key=args.api_key)
    else:
        llm = OllamaLLM(model=args.model, base_url=args.ollama_url)

    console.print("[green]✓ 파이프라인 준비 완료![/green]\n")

    return RAGPipeline(
        document_processor=processor,
        embedding=embedding,
        vector_db=vector_db,
        llm=llm,
    )


def cmd_ingest(pipeline: RAGPipeline, path: str):
    """Ingest documents into the pipeline."""
    path_obj = Path(path)

    if path_obj.is_file():
        console.print(f"[cyan]📥 문서 추가 중: {path_obj.name}[/cyan]")
        try:
            chunks = pipeline.ingest_document(str(path_obj))
            console.print(f"[green]✓ {chunks}개의 청크가 추가되었습니다.[/green]")
        except Exception as e:
            console.print(f"[red]❌ 오류: {e}[/red]")
    elif path_obj.is_dir():
        console.print(f"[cyan]📥 디렉토리에서 문서 추가 중: {path}[/cyan]")
        try:
            chunks = pipeline.ingest_directory(str(path_obj))
            console.print(f"[green]✓ 총 {chunks}개의 청크가 추가되었습니다.[/green]")
        except Exception as e:
            console.print(f"[red]❌ 오류: {e}[/red]")
    else:
        console.print(f"[red]❌ 파일 또는 디렉토리를 찾을 수 없습니다: {path}[/red]")


def cmd_query(pipeline: RAGPipeline, question: str, k: int = 5):
    """Query the pipeline and display answer."""
    console.print(f"\n[bold cyan]🔍 질문:[/bold cyan] {question}\n")

    with console.status("[bold green]답변 생성 중...[/bold green]"):
        answer = pipeline.query(question, k=k)

    console.print(Panel(Markdown(answer), title="📝 답변", border_style="green"))


def cmd_search(pipeline: RAGPipeline, query: str, k: int = 5):
    """Search for relevant documents."""
    console.print(f"\n[bold cyan]🔍 검색어:[/bold cyan] {query}\n")

    with console.status("[bold green]검색 중...[/bold green]"):
        docs = pipeline.retrieve(query, k=k)

    if not docs:
        console.print("[yellow]검색 결과가 없습니다.[/yellow]")
        return

    table = Table(title="📚 검색 결과")
    table.add_column("#", style="cyan", width=3)
    table.add_column("출처", style="magenta", width=20)
    table.add_column("내용", style="white")
    table.add_column("거리", style="dim", width=8)

    for i, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("source", "Unknown")).name
        content = (
            doc.page_content[:200] + "..."
            if len(doc.page_content) > 200
            else doc.page_content
        )
        distance = (
            f"{doc.metadata.get('distance', 'N/A'):.4f}"
            if isinstance(doc.metadata.get("distance"), float)
            else "N/A"
        )
        table.add_row(str(i), source, content, distance)

    console.print(table)


def cmd_stats(pipeline: RAGPipeline):
    """Show pipeline statistics."""
    stats = pipeline.get_stats()

    table = Table(title="📊 통계")
    table.add_column("항목", style="cyan")
    table.add_column("값", style="green")

    table.add_row("저장된 문서 청크 수", str(stats["document_count"]))

    console.print(table)


def interactive_mode(pipeline: RAGPipeline, k: int = 5):
    """Run interactive chat mode."""
    console.print(
        Panel(
            "[bold]TL.DR RAG 시스템에 오신 것을 환영합니다![/bold]\n\n"
            "사용 가능한 명령어:\n"
            "  [cyan]/add <파일경로>[/cyan] - 문서 추가\n"
            "  [cyan]/search <검색어>[/cyan] - 문서 검색\n"
            "  [cyan]/stats[/cyan] - 통계 보기\n"
            "  [cyan]/help[/cyan] - 도움말\n"
            "  [cyan]/quit[/cyan] - 종료\n\n"
            "질문을 입력하면 RAG 기반 답변을 생성합니다.",
            title="🚀 TL.DR",
            border_style="blue",
        )
    )

    stats = pipeline.get_stats()
    console.print(f"[dim]현재 저장된 문서: {stats['document_count']}개 청크[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]")

            if not user_input.strip():
                continue

            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if command in ("/quit", "/exit", "/q"):
                    console.print("[yellow]👋 안녕히 가세요![/yellow]")
                    break

                if command == "/add":
                    if arg:
                        cmd_ingest(pipeline, arg)
                    else:
                        console.print("[red]사용법: /add <파일경로>[/red]")
                elif command == "/search":
                    if arg:
                        cmd_search(pipeline, arg, k)
                    else:
                        console.print("[red]사용법: /search <검색어>[/red]")
                elif command == "/stats":
                    cmd_stats(pipeline)
                elif command == "/help":
                    console.print(
                        "\n[bold]명령어 목록:[/bold]\n"
                        "  /add <파일경로> - PDF 문서를 벡터 DB에 추가\n"
                        "  /search <검색어> - 유사한 문서 청크 검색\n"
                        "  /stats - 현재 저장된 문서 통계\n"
                        "  /quit - 프로그램 종료\n\n"
                        "일반 텍스트를 입력하면 RAG 기반 질의응답을 수행합니다."
                    )
                else:
                    console.print(f"[red]알 수 없는 명령어: {command}[/red]")
            else:
                cmd_query(pipeline, user_input, k)

        except KeyboardInterrupt:
            console.print("\n[yellow]👋 안녕히 가세요![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]오류가 발생했습니다: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(
        description="TL.DR - PDF 문서 기반 RAG 질의응답 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 문서 추가 후 대화형 모드 (기본: Ollama)
  python -m src.cli --add /path/to/document.pdf

  # Ollama 다른 모델 사용
  python -m src.cli --model mistral

  # OpenAI 사용 (API 키 필요)
  python -m src.cli --llm openai --model gpt-4o

  # 직접 질문
  python -m src.cli --question "문서 내용에 대해 설명해주세요"
        """,
    )

    # Document processing
    parser.add_argument(
        "--add",
        "-a",
        metavar="PATH",
        help="추가할 문서 파일 또는 디렉토리 경로",
    )
    parser.add_argument(
        "--processor",
        "-p",
        choices=["pymupdf", "unstructured"],
        default="pymupdf",
        help="문서 처리기 (기본값: pymupdf)",
    )

    # Embedding
    parser.add_argument(
        "--embedding",
        "-e",
        choices=EmbeddingFactory.get_available_embeddings(),
        default="bge-m3",
        help="임베딩 모델 (기본값: bge-m3)",
    )

    # Vector DB
    parser.add_argument(
        "--db-path",
        default="./chroma_db",
        help="ChromaDB 저장 경로 (기본값: ./chroma_db)",
    )
    parser.add_argument(
        "--collection",
        default="tldr_documents",
        help="ChromaDB 컬렉션 이름 (기본값: tldr_documents)",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        default=True,
        help="DB를 디스크에 저장 (기본값: True)",
    )
    parser.add_argument(
        "--no-persist",
        action="store_false",
        dest="persist",
        help="메모리 전용 모드 (종료 시 데이터 삭제)",
    )

    # LLM
    parser.add_argument(
        "--llm",
        "-l",
        choices=["openai", "ollama"],
        default="ollama",
        help="LLM 백엔드 (기본값: ollama)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="LLM 모델 (기본값 - ollama: llama3.2, openai: gpt-4o-mini)",
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API 키 (또는 OPENAI_API_KEY 환경변수 사용)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama 서버 URL (기본값: http://localhost:11434)",
    )

    # Query
    parser.add_argument(
        "--question",
        "-q",
        help="질문 (비대화형 모드)",
    )
    parser.add_argument(
        "--search",
        "-s",
        help="문서 검색 (비대화형 모드)",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="검색할 문서 수 (기본값: 5)",
    )

    args = parser.parse_args()

    # Set default model based on LLM backend
    if args.model is None:
        args.model = "gpt-4o-mini" if args.llm == "openai" else "llama3.2"

    console.print(
        Panel(
            "[bold blue]TL.DR[/bold blue] - [dim]Too Long; Didn't Read[/dim]\n"
            "PDF 문서 기반 RAG 질의응답 시스템",
            border_style="blue",
        )
    )

    # Create pipeline
    pipeline = create_pipeline(args)

    # Handle commands
    if args.add:
        cmd_ingest(pipeline, args.add)

    if args.search:
        cmd_search(pipeline, args.search, args.k)
    elif args.question:
        cmd_query(pipeline, args.question, args.k)
    else:
        # Interactive mode
        interactive_mode(pipeline, args.k)


if __name__ == "__main__":
    main()
