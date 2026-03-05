import argparse
from scripts.llm.llm_comprehensive_evaluation import llm_phenotyping_async, entropy_evaluation_async
from scripts.constants import (MODELS, PHENOTYPES, CHUNKING_METHODS, PROMPT_STRATEGIES, AGGREGATION_METHODS,
                               MODELS_DEBUG, PHENOTYPES_DEBUG, CHUNKING_METHODS_DEBUG, PROMPT_STRATEGIES_DEBUG, AGGREGATION_METHODS_DEBUG,
                               MODELS_TOY, PHENOTYPES_TOY, CHUNKING_METHODS_TOY, PROMPT_STRATEGIES_TOY, AGGREGATION_METHODS_TOY)



def create_parser():
    parser = argparse.ArgumentParser(description="Run LLM Phenotyping and Entropy Evaluation Tools")

    # General options
    parser.add_argument('--run_llm_phenotyping', action='store_true', help='Run the LLM phenotyping function')
    parser.add_argument('--run_entropy', action='store_true', help='Run the Entropy evaluation function')

    # LLM phenotyping function arguments
    parser.add_argument('--run_parsing', action='store_true', default=True, help='Run parsing in LLM phenotyping')
    parser.add_argument('--run_evaluation', action='store_true', default=True, help='Run evaluation in LLM phenotyping')
    parser.add_argument('--run_json_parsing_error_analysis', action='store_true', default=True,
                        help='Run JSON parsing error analysis in LLM phenotyping')
    parser.add_argument('--show_overall_bar', action='store_true', default=True, help='Show overall progress bar in LLM phenotyping')
    parser.add_argument('--show_config_bar', action='store_true', default=True,
                        help='Show configuration progress bar in LLM phenotyping')
    parser.add_argument('--show_chunk_bar', action='store_false', default=True, help='Show chunk progress bar in LLM phenotyping')
    parser.add_argument('--etp_cutoff', type=int, default=2, help='ETP cutoff value for LLM phenotyping')

    # Consistency function arguments
    parser.add_argument('--k', type=int, default=3, help='Number of iterations for consistency evaluation')

    # Common arguments
    parser.add_argument('--models', nargs='+', default=MODELS, help='List of models')
    parser.add_argument('--phenotypes', nargs='+', default=PHENOTYPES, help='List of phenotypes')
    parser.add_argument('--chunking_methods', nargs='+', default=CHUNKING_METHODS, help='List of chunking methods')
    parser.add_argument('--prompt_strategies', nargs='+', default=PROMPT_STRATEGIES, help='List of prompt strategies')
    parser.add_argument('--aggregation_methods', nargs='+', default=AGGREGATION_METHODS, help='List of aggregation methods')

    return parser


def main_cli():
    parser = create_parser()
    args = parser.parse_args()

    if args.run_llm_phenotyping:
        llm_phenotyping_async(run_parsing=args.run_parsing, run_evaluation=args.run_evaluation,
                        run_json_parsing_error_analysis=args.run_json_parsing_error_analysis,
                        show_overall_bar=args.show_overall_bar, show_config_bar=args.show_config_bar,
                        show_chunk_bar=args.show_chunk_bar, etp_cutoff=args.etp_cutoff,
                        phenotypes=args.phenotypes, models=args.models,
                        chunking_methods=args.chunking_methods, prompt_strategies=args.prompt_strategies,
                        aggregation_methods=args.aggregation_methods
                        )

    if args.run_entropy:
        entropy_evaluation_async(phenotypes=args.phenotypes, models=args.models,
                         chunking_methods=args.chunking_methods, prompt_strategies=args.prompt_strategies, k=args.k)


if __name__ == "__main__":
    main_cli()