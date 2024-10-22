using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;
using OllamaSharp;
using OllamaSharp.Models;

namespace RetrievalAugmentedGeneration.Services;

public class CustomEmbeddingGenerationService(string modelUrl, string embeddingModelName) : ITextEmbeddingGenerationService
{
    public IReadOnlyDictionary<string, object?> Attributes => throw new NotImplementedException();

    public async Task<IList<ReadOnlyMemory<float>>> GenerateEmbeddingsAsync(IList<string> data, Kernel? kernel = null, CancellationToken cancellationToken = default)
    {
        var ollama = new OllamaApiClient(modelUrl, embeddingModelName);

        var models = (await ollama.ListLocalModels(cancellationToken)).ToList();

        if (!models.Any(model => model.Name.Contains(embeddingModelName, StringComparison.InvariantCultureIgnoreCase)))
        {
            var idx = 0;
            await foreach (var status in ollama.PullModel(embeddingModelName, cancellationToken))
            {
                if (idx % 100 == 0) Console.WriteLine($"{status!.Percent}% {status.Status}");
                idx += 1;
            }
        }

        var embedResponse = await ollama.Embed(new EmbedRequest()
        {
            Model = embeddingModelName,
            Input = [.. data],
        }, cancellationToken);

        if (embedResponse.Embeddings.Count == 0) throw new InvalidOperationException();

        var embeddings = embedResponse.Embeddings;

        return embeddings.Select(embedding =>
        {
            var asFloatArray = embedding.Select(Convert.ToSingle).ToArray();
            return new ReadOnlyMemory<float>(asFloatArray);
        }).ToList();
    }
}