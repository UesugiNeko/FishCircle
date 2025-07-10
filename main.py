import toml
import argparse
import os
import sys
from src.agents.coordinator_agent import CoordinatorAgent
from src.formats.latex.utils import get_profect_dirs, batch_download_arxiv_tex, extract_compressed_files, get_arxiv_category
from tqdm import tqdm

base_dir = os.getcwd()
sys.path.append(base_dir)


def main():
    """
    Main function to run the LaTeXTrans application.
    Allows overriding paper_list from command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.toml", help="Path to the config TOML file.")
    parser.add_argument("paper_ids", nargs="*", help="Optional list of arXiv paper IDs to override config.")
    parser.add_argument("--model", type=str, default="deepseek-v3", help="Model for translating.")
    parser.add_argument("--base_url", type=str, default="", help="Model for translating.")
    parser.add_argument("--api_key", type=str, default="", help="Model for translating.")
    # parser.add_argument("--tl", type=str, default="ch", help="Target language.")
    # parser.add_argument("--sl", type=str, default="en", help="Source language.")
    # parser.add_argument("--ut", type=str, default="", help="User's term dict.")


    args = parser.parse_args()

    args_dict = vars(args)

    config = toml.load(args.config)

    # override paper_list if user passed in IDs via CLI
    # 之后设置参数时，config和args的传递参考这个
    if args.paper_ids:
        config["paper_list"] = args.paper_ids

    paper_list = config.get("paper_list", [])
    projects_dir = os.path.join(base_dir, config.get("tex_sources_dir", "tex source"))
    output_dir = os.path.join(base_dir, config.get("output_dir", "outputs"))

    if paper_list:
        projects = batch_download_arxiv_tex(paper_list, projects_dir)
        config["category"] = get_arxiv_category(paper_list)
        extract_compressed_files(projects_dir)
    else:
        print("⚠️ No paper list provided. Using existing projects in the specified directory.")
        extract_compressed_files(projects_dir)
        projects = get_profect_dirs(projects_dir)
        if not projects:
            raise ValueError("❌ No projects found. Check 'tex_sources_dir' and 'paper_list' in config.")

    for project_dir in tqdm(projects, desc="Processing projects", unit="project"):

        try:
            LaTexTrans = CoordinatorAgent(
                config=config,
                project_dir=project_dir,
                output_dir=output_dir
            )
            LaTexTrans.workflow_latextrans()
        except Exception as e:
            print(f"❌ Error processing project {os.path.basename(project_dir)}: {e}")
            continue

if __name__ == "__main__":
    main()
