import evaluate
from bert_score import score
import pandas as pd

# ✍️ Reference summaries (human-written / gold standard)
reference_summaries = [
    "AI is revolutionizing business operations by automating tasks, improving analytics, forecasting demand, and enhancing decision-making, but adoption requires addressing privacy, bias, and cost challenges.",
    "AI enhances healthcare through early disease detection, personalized treatments, predictive analytics, and patient support, but raises privacy, bias, and ethical concerns that require proper regulation.",
    "AI is transforming society by personalizing education, improving governance, and enhancing online platforms, but challenges such as job displacement, bias, privacy, and inequality must be addressed."
]

# 🤖 Generated summaries from your summarizer
generated_summaries = [
    "Artificial Intelligence (AI) is increasingly becoming a strategic asset for businesses across the globe. By automating repetitive tasks, AI allows employees to focus on creative and high-value work. However, adoption of AI also comes with challenges, such as data privacy concerns, algorithmic bias, and the high cost of implementation. Organizations must balance innovation with ethical considerations to ensure responsible and sustainable growth.",
    "Artificial Intelligence (AI) is playing a transformative role in healthcare. Machine learning algorithms can analyze medical images to detect diseases such as cancer at earlier stages. Personalized medicine is another major area where AI tailors treatments based on genetic data, lifestyle, and medical history. Despite its benefits, the use of AI in healthcare raises significant ethical and legal concerns.",
    "Artificial Intelligence (AI) is reshaping society in profound ways, influencing how we work, learn, communicate, and make decisions. Governments are using AI for policy analysis, infrastructure planning, and public safety initiatives. Social media platforms rely on AI algorithms to filter content, detect harmful behavior, and recommend relevant posts."
]

# ✅ ROUGE Evaluation
rouge = evaluate.load("rouge")
rouge_results = rouge.compute(predictions=generated_summaries, references=reference_summaries)
print("📊 ROUGE Scores:")
for key, value in rouge_results.items():
    print(f"{key}: {value:.4f}")

# ✅ BLEU Evaluation (✅ FIXED - pass plain strings)
bleu = evaluate.load("bleu")
bleu_result = bleu.compute(
    predictions=generated_summaries,
    references=[[ref] for ref in reference_summaries]
)
print("\n📊 BLEU Score:", bleu_result["bleu"])

# ✅ BERTScore Evaluation (⚠️ This can take ~1 min)
print("\n⏳ Calculating BERTScore (this may take 30-60 seconds)...")
P, R, F1 = score(generated_summaries, reference_summaries, lang="en", verbose=True)
print("\n📊 BERTScore:")
print("Precision:", P.mean().item())
print("Recall:", R.mean().item())
print("F1:", F1.mean().item())

# ✅ Save results to CSV
results = {
    "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERT-F1"],
    "Score": [
        rouge_results["rouge1"],
        rouge_results["rouge2"],
        rouge_results["rougeL"],
        bleu_result["bleu"],
        F1.mean().item()
    ]
}

df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)
print("\n✅ Results saved to 'evaluation_results.csv'")
