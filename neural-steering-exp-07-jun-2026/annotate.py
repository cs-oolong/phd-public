#!/usr/bin/env python3
"""CLI para avaliar e anotar resultados de steering experiments."""

import os
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
SEPARATOR = "─" * 60


def get_out_files():
    """Collect all .out files sorted by path."""
    return sorted(RESULTS_DIR.rglob("*.out"))


def notes_path(out_file: Path) -> Path:
    return out_file.with_suffix(".notes")


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def show_file(path: Path):
    print(f"\n\033[1;36m{SEPARATOR}\033[0m")
    print(f"\033[1;33m 📄 {path.relative_to(RESULTS_DIR)}\033[0m")
    print(f"\033[1;36m{SEPARATOR}\033[0m\n")
    print(path.read_text())
    print(f"\n\033[1;36m{SEPARATOR}\033[0m")


def prompt_annotation() -> str | None:
    """Multi-line input. Empty line + Enter to finish, 'skip' to skip."""
    print("\n\033[1;32mEscreva suas observações (linha vazia para finalizar, 'skip' para pular):\033[0m")
    lines = []
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if line.strip().lower() == "skip":
            return None
        if line == "" and lines:
            break
        if line == "" and not lines:
            continue
        lines.append(line)
    return "\n".join(lines)


def handle_existing_notes(npath: Path) -> str | None:
    """Show existing notes and ask what to do."""
    content = npath.read_text().strip()
    if not content:
        return "new"

    print(f"\n\033[1;35m📝 Anotação existente:\033[0m")
    print(f"\033[2m{content}\033[0m\n")

    while True:
        choice = input("\033[1m[e]ditar / [a]vançar / [s]ubstituir? \033[0m").strip().lower()
        if choice in ("a", "avançar", "avancar", ""):
            return "skip"
        elif choice in ("e", "editar"):
            return "edit"
        elif choice in ("s", "substituir"):
            return "replace"
        else:
            print("  Opção inválida. Use 'e', 'a' ou 's'.")


def main():
    out_files = get_out_files()
    if not out_files:
        print("Nenhum arquivo .out encontrado em results/")
        sys.exit(1)

    total = len(out_files)
    print(f"\n\033[1mEncontrados {total} arquivos para anotar.\033[0m")
    print("Ctrl+C a qualquer momento para sair.\n")

    annotated = sum(1 for f in out_files if notes_path(f).exists() and notes_path(f).read_text().strip())
    if annotated:
        print(f"\033[2m({annotated}/{total} já possuem anotações)\033[0m\n")

    for i, out_file in enumerate(out_files, 1):
        npath = notes_path(out_file)

        clear_screen()
        print(f"\033[1m[{i}/{total}]\033[0m", end="")
        show_file(out_file)

        if npath.exists() and npath.read_text().strip():
            action = handle_existing_notes(npath)
            if action == "skip":
                continue
            elif action == "edit":
                existing = npath.read_text().strip()
                print(f"\n\033[2m(conteúdo atual será mantido, adicione abaixo)\033[0m")
                new_text = prompt_annotation()
                if new_text:
                    npath.write_text(existing + "\n" + new_text + "\n")
                    print("\033[1;32m✓ Atualizado.\033[0m")
                continue
            # "replace" falls through to fresh annotation

        annotation = prompt_annotation()
        if annotation:
            npath.write_text(annotation + "\n")
            print("\033[1;32m✓ Salvo.\033[0m")
        else:
            print("\033[2m(pulado)\033[0m")

    print(f"\n\033[1;32mDone! Todos os {total} arquivos foram revisados.\033[0m\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n\033[2mSaindo...\033[0m")
        sys.exit(0)
