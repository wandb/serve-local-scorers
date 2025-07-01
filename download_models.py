#!/usr/bin/env python3
"""
Weave Local Scorers integration for the encoder server.
Downloads and manages Weave scorer models from HuggingFace.
"""
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import click
from huggingface_hub import snapshot_download
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)


@dataclass
class WeaveScorerInfo:
    """Information about a Weave scorer model."""

    name: str
    hf_model_id: str
    task_type: str
    description: str
    size_params: str
    local_path: Optional[str] = None


class WeaveScorerManager:
    """Manager for Weave Local Scorers - download, list, and organize models."""

    # Known Weave scorers from the HF collection
    WEAVE_SCORERS = {
        "bias": WeaveScorerInfo(
            name="WeaveBiasScorerV1",
            hf_model_id="wandb/WeaveBiasScorerV1",
            task_type="text-classification",
            description="Detects bias in text related to gender, race, and origin",
            size_params="0.1B",
        ),
        "toxicity": WeaveScorerInfo(
            name="WeaveToxicityScorerV1",
            hf_model_id="wandb/WeaveToxicityScorerV1",
            task_type="text-classification",
            description="Identifies toxic content across five dimensions",
            size_params="0.1B",
        ),
        "hallucination": WeaveScorerInfo(
            name="WeaveHallucinationScorerV1",
            hf_model_id="wandb/WeaveHallucinationScorerV1",
            task_type="text-classification",
            description="Checks for hallucinations in AI system outputs",
            size_params="0.1B",
        ),
        "context_relevance": WeaveScorerInfo(
            name="WeaveContextRelevanceScorerV1",
            hf_model_id="wandb/WeaveContextRelevanceScorerV1",
            task_type="token-classification",
            description="Evaluates relevance of context in RAG systems",
            size_params="0.2B",
        ),
        "coherence": WeaveScorerInfo(
            name="WeaveCoherenceScorerV1",
            hf_model_id="wandb/WeaveCoherenceScorerV1",
            task_type="text-classification",
            description="Assesses text coherence and logical flow",
            size_params="0.1B",
        ),
        "fluency": WeaveScorerInfo(
            name="WeaveFluencyScorerV1",
            hf_model_id="wandb/WeaveFluencyScorerV1",
            task_type="text-classification",
            description="Measures text readability and natural language quality",
            size_params="0.1B",
        ),
    }

    def __init__(self, models_dir: str = "models"):
        """Initialize the scorer manager."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

    def list_available_scorers(self) -> Dict[str, WeaveScorerInfo]:
        """List all available Weave scorers."""
        return self.WEAVE_SCORERS.copy()

    def list_downloaded_scorers(self) -> List[str]:
        """List scorers that have been downloaded locally."""
        downloaded = []
        for key, scorer in self.WEAVE_SCORERS.items():
            local_path = self.models_dir / scorer.name.lower()
            if local_path.exists() and (local_path / "config.json").exists():
                downloaded.append(key)
        return downloaded

    def get_scorer_info(self, scorer_key: str) -> Optional[WeaveScorerInfo]:
        """Get information about a specific scorer."""
        return self.WEAVE_SCORERS.get(scorer_key)

    def download_scorer(self, scorer_key: str, force: bool = False) -> str:
        """Download a Weave scorer model."""
        if scorer_key not in self.WEAVE_SCORERS:
            available = list(self.WEAVE_SCORERS.keys())
            raise ValueError(f"Unknown scorer '{scorer_key}'. Available: {available}")

        scorer = self.WEAVE_SCORERS[scorer_key]
        local_path = self.models_dir / scorer.name.lower()

        # Check if already downloaded
        if local_path.exists() and not force:
            if (local_path / "config.json").exists():
                logger.info(f"Scorer '{scorer_key}' already downloaded at {local_path}")
                return str(local_path)
            else:
                logger.warning(
                    f"Incomplete download found at {local_path}, re-downloading"
                )
                shutil.rmtree(local_path)

        logger.info(f"Downloading {scorer.name} from {scorer.hf_model_id}...")

        try:
            # Download the model
            downloaded_path = snapshot_download(
                repo_id=scorer.hf_model_id,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                resume_download=True,
            )

            # Verify download
            if not (local_path / "config.json").exists():
                raise FileNotFoundError("Downloaded model is missing config.json")

            logger.info(f"Successfully downloaded {scorer.name} to {local_path}")
            return str(local_path)

        except Exception as e:
            logger.error(f"Failed to download {scorer.name}: {e}")
            if local_path.exists():
                shutil.rmtree(local_path)
            raise

    def download_all_scorers(self, force: bool = False) -> Dict[str, str]:
        """Download all available Weave scorers."""
        results = {}
        failed = []

        for scorer_key in self.WEAVE_SCORERS:
            try:
                path = self.download_scorer(scorer_key, force=force)
                results[scorer_key] = path
                logger.info(f"✅ Downloaded {scorer_key}")
            except Exception as e:
                logger.error(f"❌ Failed to download {scorer_key}: {e}")
                failed.append(scorer_key)

        if failed:
            logger.warning(f"Failed to download: {failed}")

        logger.info(f"Downloaded {len(results)}/{len(self.WEAVE_SCORERS)} scorers")
        return results

    def get_model_path(self, scorer_key: str) -> str:
        """Get the local path for a scorer model."""
        if scorer_key not in self.WEAVE_SCORERS:
            raise ValueError(f"Unknown scorer '{scorer_key}'")

        scorer = self.WEAVE_SCORERS[scorer_key]
        local_path = self.models_dir / scorer.name.lower()

        if not local_path.exists() or not (local_path / "config.json").exists():
            raise FileNotFoundError(
                f"Scorer '{scorer_key}' not found locally. "
                f"Download it first with: download_scorer('{scorer_key}')"
            )

        return str(local_path)

    def verify_model(self, scorer_key: str) -> bool:
        """Verify that a model is properly downloaded and valid."""
        try:
            model_path = self.get_model_path(scorer_key)
            model_dir = Path(model_path)

            # Check for required files
            required_files = ["config.json"]
            optional_files = [
                "model.safetensors",
                "pytorch_model.bin",
                "tokenizer.json",
            ]

            missing_required = [
                f for f in required_files if not (model_dir / f).exists()
            ]
            if missing_required:
                logger.error(f"Missing required files: {missing_required}")
                return False

            # Check if at least one model file exists
            has_model_file = any((model_dir / f).exists() for f in optional_files)
            if not has_model_file:
                logger.error("No model weights file found")
                return False

            logger.info(f"✅ Model '{scorer_key}' is valid")
            return True

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

    def get_model_summary(self) -> Dict[str, Dict[str, Union[str, bool]]]:
        """Get a summary of all models and their status."""
        summary = {}

        for key, scorer in self.WEAVE_SCORERS.items():
            local_path = self.models_dir / scorer.name.lower()
            is_downloaded = (
                local_path.exists() and (local_path / "config.json").exists()
            )
            is_valid = self.verify_model(key) if is_downloaded else False

            summary[key] = {
                "name": scorer.name,
                "description": scorer.description,
                "task_type": scorer.task_type,
                "size": scorer.size_params,
                "hf_model_id": scorer.hf_model_id,
                "downloaded": is_downloaded,
                "valid": is_valid,
                "local_path": str(local_path) if is_downloaded else None,
            }

        return summary


def create_interactive_cli():
    """Create beautiful interactive CLI using Rich."""
    console = Console()

    def show_banner():
        """Display beautiful banner."""
        banner_text = Text()
        banner_text.append("🌊 ", style="cyan bold")
        banner_text.append("Weave Local Scorers", style="bold blue")
        banner_text.append(" Manager", style="bold white")

        description = Text()
        description.append(
            "Download and manage AI scoring models from the Weave collection",
            style="dim",
        )

        panel = Panel(
            Align.center(banner_text + "\n" + description),
            box=box.DOUBLE,
            border_style="blue",
            padding=(1, 2),
        )
        console.print(panel)
        console.print()

    def show_scorer_table(manager: WeaveScorerManager):
        """Display beautiful table of available scorers."""
        table = Table(
            title="📊 Available Weave Scorers",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Status", justify="center", style="bold", min_width=8)
        table.add_column("Scorer", style="bold cyan", min_width=15)
        table.add_column("Description", style="white", min_width=40)
        table.add_column("Type", style="yellow", min_width=15)
        table.add_column("Size", justify="right", style="green")

        summary = manager.get_model_summary()

        for key, info in summary.items():
            # Status with emoji and color
            if info["downloaded"] and info["valid"]:
                status = "✅ Ready"
                status_style = "green"
            elif info["downloaded"]:
                status = "⚠️ Invalid"
                status_style = "yellow"
            else:
                status = "📥 Available"
                status_style = "blue"

            # Task type with emoji
            task_emoji = "🏷️" if info["task_type"] == "text-classification" else "🔤"
            task_display = f"{task_emoji} {info['task_type']}"

            table.add_row(
                f"[{status_style}]{status}[/{status_style}]",
                key,
                info["description"],
                task_display,
                info["size"],
            )

        console.print(table)
        console.print()

    def show_selection_grid(manager: WeaveScorerManager):
        """Show scorers in a grid format for selection."""
        available_scorers = manager.list_available_scorers()
        downloaded = manager.list_downloaded_scorers()

        scorer_panels = []
        for key, scorer in available_scorers.items():
            is_downloaded = key in downloaded

            # Create status indicator
            if is_downloaded:
                status_text = Text("✅ Downloaded", style="bold green")
            else:
                status_text = Text("📥 Available", style="bold blue")

            # Create content
            content = Text()
            content.append(f"{key}\n", style="bold cyan")
            content.append(f"{scorer.description}\n", style="white")
            content.append(f"Size: {scorer.size_params}", style="dim")

            # Create panel
            panel = Panel(
                content + "\n" + status_text,
                title=f"[bold]{scorer.name}[/bold]",
                border_style="green" if is_downloaded else "blue",
                padding=(1, 1),
                width=35,
            )
            scorer_panels.append(panel)

        # Display in columns
        console.print(Columns(scorer_panels, equal=True, expand=True))
        console.print()

    def download_with_progress(
        manager: WeaveScorerManager, scorers_to_download: list, force: bool = False
    ):
        """Download scorers with beautiful progress display."""
        if not scorers_to_download:
            console.print("[yellow]⚠️ No scorers selected for download.[/yellow]")
            return

        console.print(
            f"[bold green]🚀 Starting download of {len(scorers_to_download)} scorer(s)...[/bold green]\n"
        )

        success_count = 0
        failed_scorers = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=False,
        ) as progress:
            overall_task = progress.add_task(
                f"[bold blue]Downloading {len(scorers_to_download)} scorers...",
                total=len(scorers_to_download),
            )

            for i, scorer_key in enumerate(scorers_to_download):
                scorer_info = manager.get_scorer_info(scorer_key)

                # Create task for this scorer
                scorer_task = progress.add_task(
                    f"[cyan]Downloading {scorer_info.name}...", total=100
                )

                try:
                    # Update progress
                    progress.update(scorer_task, advance=25)
                    path = manager.download_scorer(scorer_key, force=force)
                    progress.update(scorer_task, advance=75)

                    # Mark as complete
                    progress.update(
                        scorer_task,
                        completed=100,
                        description=f"[green]✅ {scorer_info.name}",
                    )
                    success_count += 1

                except Exception as e:
                    progress.update(
                        scorer_task,
                        completed=100,
                        description=f"[red]❌ {scorer_info.name} - {str(e)[:30]}...",
                    )
                    failed_scorers.append(scorer_key)

                # Update overall progress
                progress.update(overall_task, advance=1)

        # Show summary
        console.print()
        if success_count > 0:
            console.print(
                f"[bold green]🎉 Successfully downloaded {success_count} scorer(s)![/bold green]"
            )

        if failed_scorers:
            console.print(
                f"[bold red]❌ Failed to download: {', '.join(failed_scorers)}[/bold red]"
            )

        console.print()

    def interactive_scorer_selection(manager: WeaveScorerManager):
        """Interactive scorer selection interface."""
        available_scorers = list(manager.list_available_scorers().keys())
        downloaded = manager.list_downloaded_scorers()
        not_downloaded = [s for s in available_scorers if s not in downloaded]

        if not not_downloaded:
            console.print("[green]🎉 All scorers are already downloaded![/green]")
            return []

        console.print("[bold]Select scorers to download:[/bold]")
        console.print(
            "[dim]Enter numbers separated by commas (e.g., 1,3,5) or 'all' for all scorers[/dim]\n"
        )

        # Show numbered list
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("", justify="right", style="dim", width=3)
        table.add_column("Scorer", style="bold cyan")
        table.add_column("Description", style="white")
        table.add_column("Size", style="green")
        table.add_column("Status", style="yellow")

        for i, scorer_key in enumerate(available_scorers, 1):
            scorer_info = manager.get_scorer_info(scorer_key)
            status = "Downloaded" if scorer_key in downloaded else "Available"
            status_style = "green" if scorer_key in downloaded else "blue"

            table.add_row(
                str(i),
                scorer_key,
                scorer_info.description[:50] + "..."
                if len(scorer_info.description) > 50
                else scorer_info.description,
                scorer_info.size_params,
                f"[{status_style}]{status}[/{status_style}]",
            )

        console.print(table)
        console.print()

        while True:
            choice = (
                Prompt.ask("[bold]Your selection", default="all", show_default=True)
                .strip()
                .lower()
            )

            if choice == "all":
                return not_downloaded
            elif choice == "none" or choice == "":
                return []
            else:
                try:
                    # Parse comma-separated numbers
                    selected_indices = [int(x.strip()) for x in choice.split(",")]
                    selected_scorers = []

                    for idx in selected_indices:
                        if 1 <= idx <= len(available_scorers):
                            scorer_key = available_scorers[idx - 1]
                            if scorer_key not in downloaded:
                                selected_scorers.append(scorer_key)
                            else:
                                console.print(
                                    f"[yellow]⚠️ Scorer '{scorer_key}' is already downloaded[/yellow]"
                                )
                        else:
                            console.print(f"[red]❌ Invalid number: {idx}[/red]")

                    if selected_scorers:
                        return selected_scorers
                    else:
                        console.print("[yellow]⚠️ No valid scorers selected[/yellow]")

                except ValueError:
                    console.print(
                        "[red]❌ Invalid input. Please enter numbers separated by commas or 'all'[/red]"
                    )

    return {
        "show_banner": show_banner,
        "show_scorer_table": show_scorer_table,
        "show_selection_grid": show_selection_grid,
        "download_with_progress": download_with_progress,
        "interactive_scorer_selection": interactive_scorer_selection,
        "console": console,
    }


def main():
    """Enhanced CLI interface with beautiful Rich UI."""

    cli_components = create_interactive_cli()
    console = cli_components["console"]

    @click.group(invoke_without_command=True)
    @click.option("--models-dir", default="models", help="Directory to store models")
    @click.pass_context
    def cli(ctx, models_dir):
        """🌊 Weave Local Scorers Manager - Beautiful CLI for AI model management"""
        ctx.ensure_object(dict)
        ctx.obj["models_dir"] = models_dir
        ctx.obj["manager"] = WeaveScorerManager(models_dir)

        if ctx.invoked_subcommand is None:
            # Interactive mode
            interactive_main(ctx.obj["manager"])

    @cli.command()
    @click.pass_context
    def interactive(ctx):
        """🎯 Interactive mode for downloading scorers"""
        interactive_main(ctx.obj["manager"])

    @cli.command()
    @click.option("--downloaded-only", is_flag=True, help="Show only downloaded models")
    @click.pass_context
    def list(ctx, downloaded_only):
        """📋 List available scorers"""
        manager = ctx.obj["manager"]
        cli_components["show_banner"]()

        if downloaded_only:
            downloaded = manager.list_downloaded_scorers()
            if downloaded:
                console.print(
                    f"[bold green]Downloaded scorers ({len(downloaded)}):[/bold green]\n"
                )
                for key in downloaded:
                    scorer = manager.get_scorer_info(key)
                    console.print(
                        f"  [green]✅[/green] [bold cyan]{key}[/bold cyan]: {scorer.description}"
                    )
            else:
                console.print("[yellow]No scorers downloaded yet.[/yellow]")
        else:
            cli_components["show_scorer_table"](manager)

    @cli.command()
    @click.argument("scorer", required=False)
    @click.option("--force", is_flag=True, help="Force re-download")
    @click.pass_context
    def download(ctx, scorer, force):
        """📥 Download specific scorer(s) or use interactive mode"""
        manager = ctx.obj["manager"]

        if scorer:
            if scorer == "all":
                available = list(manager.list_available_scorers().keys())
                cli_components["download_with_progress"](manager, available, force)
            else:
                if scorer in manager.WEAVE_SCORERS:
                    cli_components["download_with_progress"](manager, [scorer], force)
                else:
                    console.print(f"[red]❌ Unknown scorer: {scorer}[/red]")
                    console.print(
                        f"[dim]Available: {', '.join(manager.WEAVE_SCORERS.keys())}[/dim]"
                    )
        else:
            # Interactive mode
            interactive_main(ctx.obj["manager"])

    @cli.command()
    @click.argument("scorer", required=False)
    @click.pass_context
    def verify(ctx, scorer):
        """🔍 Verify downloaded models"""
        manager = ctx.obj["manager"]

        if scorer == "all" or not scorer:
            downloaded = manager.list_downloaded_scorers()
            if not downloaded:
                console.print("[yellow]No scorers downloaded to verify.[/yellow]")
                return

            console.print("[bold]Verifying downloaded scorers...[/bold]\n")
            for key in downloaded:
                is_valid = manager.verify_model(key)
                status = (
                    "[green]✅ Valid[/green]" if is_valid else "[red]❌ Invalid[/red]"
                )
                console.print(f"  {status} [cyan]{key}[/cyan]")
        else:
            if scorer not in manager.WEAVE_SCORERS:
                console.print(f"[red]❌ Unknown scorer: {scorer}[/red]")
                return

            try:
                is_valid = manager.verify_model(scorer)
                status = (
                    "[green]✅ Valid[/green]" if is_valid else "[red]❌ Invalid[/red]"
                )
                console.print(f"{status} [cyan]{scorer}[/cyan]")
            except FileNotFoundError:
                console.print(f"[yellow]⚠️ Scorer '{scorer}' not downloaded[/yellow]")

    @cli.command()
    @click.argument("scorer")
    @click.pass_context
    def info(ctx, scorer):
        """ℹ️ Show detailed information about a scorer"""
        from rich.panel import Panel
        from rich.text import Text

        manager = ctx.obj["manager"]

        if scorer not in manager.WEAVE_SCORERS:
            console.print(f"[red]❌ Unknown scorer: {scorer}[/red]")
            return

        scorer_info = manager.get_scorer_info(scorer)
        summary = manager.get_model_summary()[scorer]

        # Create beautiful info panel
        info_text = Text()
        info_text.append(f"Name: ", style="bold")
        info_text.append(f"{scorer_info.name}\n", style="cyan")
        info_text.append(f"Description: ", style="bold")
        info_text.append(f"{scorer_info.description}\n", style="white")
        info_text.append(f"Task Type: ", style="bold")
        info_text.append(f"{scorer_info.task_type}\n", style="yellow")
        info_text.append(f"Size: ", style="bold")
        info_text.append(f"{scorer_info.size_params}\n", style="green")
        info_text.append(f"HuggingFace ID: ", style="bold")
        info_text.append(f"{scorer_info.hf_model_id}\n", style="blue")
        info_text.append(f"Downloaded: ", style="bold")
        info_text.append(
            f"{'Yes' if summary['downloaded'] else 'No'}\n",
            style="green" if summary["downloaded"] else "red",
        )

        if summary["downloaded"]:
            info_text.append(f"Local Path: ", style="bold")
            info_text.append(f"{summary['local_path']}\n", style="dim")
            info_text.append(f"Valid: ", style="bold")
            info_text.append(
                f"{'Yes' if summary['valid'] else 'No'}",
                style="green" if summary["valid"] else "red",
            )

        panel = Panel(
            info_text,
            title=f"[bold]📊 {scorer.title()} Scorer Information[/bold]",
            border_style="blue",
            padding=(1, 2),
        )
        console.print(panel)

    def interactive_main(manager):
        """Main interactive interface."""
        from rich.panel import Panel
        from rich.prompt import Confirm, Prompt
        from rich.text import Text

        cli_components["show_banner"]()
        cli_components["show_scorer_table"](manager)

        # Show quick stats
        total_scorers = len(manager.list_available_scorers())
        downloaded_count = len(manager.list_downloaded_scorers())

        stats_text = Text()
        stats_text.append(f"📊 Status: ", style="bold")
        stats_text.append(
            f"{downloaded_count}/{total_scorers} downloaded", style="cyan"
        )

        console.print(Panel(stats_text, border_style="dim"))
        console.print()

        # Ask what to do
        action = Prompt.ask(
            "[bold]What would you like to do?",
            choices=["download", "verify", "list", "exit"],
            default="download",
        )

        if action == "download":
            selected_scorers = cli_components["interactive_scorer_selection"](manager)
            if selected_scorers:
                force = Confirm.ask(
                    f"Force re-download if already exists?", default=False
                )
                cli_components["download_with_progress"](
                    manager, selected_scorers, force
                )

        elif action == "verify":
            downloaded = manager.list_downloaded_scorers()
            if downloaded:
                console.print("[bold]Verifying downloaded scorers...[/bold]\n")
                for key in downloaded:
                    is_valid = manager.verify_model(key)
                    status = (
                        "[green]✅ Valid[/green]"
                        if is_valid
                        else "[red]❌ Invalid[/red]"
                    )
                    console.print(f"  {status} [cyan]{key}[/cyan]")
            else:
                console.print("[yellow]No scorers downloaded to verify.[/yellow]")

        elif action == "list":
            cli_components["show_selection_grid"](manager)

        console.print(
            f"\n[dim]💡 Tip: Use '[bold]python download_models.py --help[/bold]' to see all available commands[/dim]"
        )

    # Setup logging to be less verbose in interactive mode
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    cli()


if __name__ == "__main__":
    main()
