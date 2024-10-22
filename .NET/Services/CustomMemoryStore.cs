using System.Dynamic;
using System.Globalization;
using System.Runtime.CompilerServices;
using Microsoft.SemanticKernel.Connectors.Chroma;
using Microsoft.SemanticKernel.Memory;
using MR = Microsoft.SemanticKernel.Memory.MemoryRecord;

namespace RetrievalAugmentedGeneration.Services;

public class CustomMemoryStore(string chromaUrl) : IMemoryStore
{
    private readonly IChromaClient _client = new ChromaClient(chromaUrl);

    public async Task CreateCollectionAsync(string collectionName, CancellationToken cancellationToken = default)
    {
        await _client.CreateCollectionAsync(collectionName, cancellationToken);
    }

    public async Task DeleteCollectionAsync(string collectionName, CancellationToken cancellationToken = default)
    {
        await _client.DeleteCollectionAsync(collectionName, cancellationToken);
    }

    public async Task<bool> DoesCollectionExistAsync(string collectionName, CancellationToken cancellationToken = default)
    {
        return (await _client.GetCollectionAsync(collectionName, cancellationToken)) != null;
    }

    public Task<MemoryRecord?> GetAsync(string collectionName, string key, bool withEmbedding = false, CancellationToken cancellationToken = default)
    {
        throw new NotImplementedException();
    }

    public IAsyncEnumerable<MemoryRecord> GetBatchAsync(string collectionName, IEnumerable<string> keys, bool withEmbeddings = false, CancellationToken cancellationToken = default)
    {
        throw new NotImplementedException();
    }

    public IAsyncEnumerable<string> GetCollectionsAsync(CancellationToken cancellationToken = default)
    {
        return _client.ListCollectionsAsync(cancellationToken);
    }

    public async Task<(MR, double)?> GetNearestMatchAsync(string collectionName, ReadOnlyMemory<float> embedding, double minRelevanceScore = 0, bool withEmbedding = false, CancellationToken cancellationToken = default)
    {
        var collectionId = await GetCollectionId(collectionName, cancellationToken);

        try
        {
            // include should be: "documents", "embeddings", "metadatas", "distances", "uris" or "data"
            var queryResult = await _client.QueryEmbeddingsAsync(collectionId, [embedding], 1, include: ["metadatas", "documents", "distances"], cancellationToken: cancellationToken);

            var record = MR.LocalRecord(
                queryResult.Ids[0][0],
                text: queryResult.Metadatas[0][0]["text"].ToString(),
                description: queryResult.Metadatas[0][0]["description"].ToString(),
                embedding: null
            );

        return (record, queryResult.Distances[0][0]);
        }
        catch (System.Exception ex)
        {
            Console.WriteLine("Error!", ex.Message);
            throw;
        }
    }

    public async IAsyncEnumerable<(MemoryRecord, double)> GetNearestMatchesAsync(string collectionName, ReadOnlyMemory<float> embedding, int limit, double minRelevanceScore = 0, bool withEmbeddings = false, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var collectionId = await GetCollectionId(collectionName, cancellationToken);
        var queryResultPerEmbedding = await _client.QueryEmbeddingsAsync(collectionId, [embedding], nResults: limit, include: ["metadatas", "documents", "distances"], cancellationToken: cancellationToken);

        foreach(var (ids, metadatas, distances) in queryResultPerEmbedding.Ids.Zip(queryResultPerEmbedding.Metadatas, queryResultPerEmbedding.Distances))
        {
            foreach (var (id, metadata, distance) in ids.Zip(metadatas, distances))
            {
                var record = MR.LocalRecord(
                    id,
                    text: metadata["text"]!.ToString(),
                    description: metadata["description"]!.ToString(),
                    embedding
                );
                yield return (record, distance);
            }
        }
    }

    public Task RemoveAsync(string collectionName, string key, CancellationToken cancellationToken = default)
    {
        throw new NotImplementedException();
    }

    public Task RemoveBatchAsync(string collectionName, IEnumerable<string> keys, CancellationToken cancellationToken = default)
    {
        throw new NotImplementedException();
    }

    public async Task<string> UpsertAsync(string collectionName, MemoryRecord record, CancellationToken cancellationToken = default)
    {
        await _client.UpsertEmbeddingsAsync(
            collectionId: collectionName,
            ids: ["0"],
            embeddings: [record.Embedding],
            cancellationToken: cancellationToken
        );

        return "0";
    }

    public async IAsyncEnumerable<string> UpsertBatchAsync(string collectionName, IEnumerable<MemoryRecord> records, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var collectionId = await GetCollectionId(collectionName, cancellationToken);
        foreach (var record in records)
        {
            try
            {
                await _client.UpsertEmbeddingsAsync(
                    collectionId,
                    ids: [record.Metadata.Id],
                    embeddings: [record.Embedding],
                    metadatas: [record.Metadata],
                    cancellationToken: cancellationToken
                );
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error!", ex.Message);
                throw;
            }
            yield return record.Metadata.Id;
        }
    }

    private async Task<string> GetCollectionId(string collectionName, CancellationToken cancellationToken = default) =>
        (await _client.GetCollectionAsync(collectionName, cancellationToken))?.Id
        ??
        throw new KeyNotFoundException($"Couldn't find a collection named {collectionName}");
} 