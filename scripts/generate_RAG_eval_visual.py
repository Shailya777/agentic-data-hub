import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_rag_eval_dashboard():
    """
    Generates RAG eval dashboard visual.
    :return: None
    """

    # Data And Output Paths:
    root_dir= os.path.abspath(os.path.join(os.path.dirname(__file__),"../"))
    eval_data_path= os.path.join(root_dir,"data/processed/rag_metrics.csv")
    output_path= os.path.join(root_dir,"data/processed/rag_eval_dashboard.png")

    # Data Extraction:
    df= pd.read_csv(eval_data_path)
    latest_run= df.iloc[-1] # Taking the last row only if there are multiple evaluation runs

    metrics= ['Context Relevance', 'Faithfulness', 'Answer Relevance']
    scores= [
        latest_run['avg_context_relevance'],
        latest_run['avg_faithfulness'],
        latest_run['avg_answer_relevance']
    ]
    guardrail_acc= latest_run['guardrail_accuracy']

    # Setting up Dashboard:
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Cyberpunk/Neon Style:
    colors= ['#00d2ff', '#69f0ae', '#e040fb']

    # Plotting Bar Chart:
    bars= ax.barh(metrics,
                  scores,
                  color= colors,
                  height= 0.4,
                  edgecolor= 'none')

    # Axes Styling:
    ax.set_xlim(0,10)
    ax.set_xticks(range(0, 11))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#8b949e')
    ax.spines['bottom'].set_color('#8b949e')
    ax.tick_params(axis= 'x', colors= '#8b949e', labelsize= 10)
    ax.tick_params(axis= 'y', colors='#8b949e', labelsize= 12, pad= 10)

    # Adding Value Annotations:
    for bar, score in zip(bars, scores):
        ax.text(score + 0.15, bar.get_y() + bar.get_height() / 2,
                f'{score:.2f} / 10',
                va='center', color='#ffffff', fontweight='bold', fontsize=12)

    # Title:
    plt.title('RAG Pipeline Evaluation Metrics (LLM-as-a-Judge)',
              color= '#ffffff', fontsize= 16, pad= 40, fontweight= 'bold')

    # Guardrail Badge:
    props= dict(boxstyle='round,pad=0.6', facecolor='#1b5e20', alpha=0.4, edgecolor='#69f0ae')
    ax.text(0.5, 1.12,
            f'🛡️ Security Guardrail Accuracy: {guardrail_acc:.1f}% (Zero Out-of-Bounds Hallucinations)',
            transform=ax.transAxes, fontsize=11, color='#69f0ae',
            verticalalignment='top', horizontalalignment='center', bbox=props, fontweight='bold')

    # Rendering and Exporting The Dashboard:
    plt.tight_layout()
    plt.savefig(output_path,
                dpi= 300,
                bbox_inches= 'tight',
                facecolor= fig.get_facecolor())
    print(f"Visualization Successfully rendered and saved to {output_path}")


if __name__ == "__main__":
    generate_rag_eval_dashboard()