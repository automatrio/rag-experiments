using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Chroma;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Memory;
using Microsoft.SemanticKernel.Plugins.Memory;
using Microsoft.SemanticKernel.Text;
using RetrievalAugmentedGeneration.Services;
using MR = Microsoft.SemanticKernel.Memory.MemoryRecord;

const string OLLAMA_URL = "http://localhost:11434";
const string CHROMA_URL = "http://localhost:8000";
const string MODEL_NAME = "llama3.2:1b";
const string EMBEDDING_MODEL_NAME = "all-minilm";

// Instantiate services.
var embeddingService = new CustomEmbeddingGenerationService(OLLAMA_URL, EMBEDDING_MODEL_NAME);
var store = new CustomMemoryStore(CHROMA_URL);
var ollamaChat = new CustomChatCompletionService(
    modelUrl: OLLAMA_URL,
    modelName: MODEL_NAME
);

// This won't do anything at the moment, but it's meant to create a memory plugin.
var memoryWithChroma = new MemoryBuilder()
    .WithChromaMemoryStore(CHROMA_URL)
    .WithMemoryStore(store)
    .WithTextEmbeddingGeneration(embeddingService)
    .Build();

// Create the SemanticKernel.
var builder = Kernel.CreateBuilder();
builder.Services.AddKeyedSingleton<IChatCompletionService>("ollamaChat", ollamaChat);
builder.Plugins.AddFromObject(new TextMemoryPlugin(memoryWithChroma));
var kernel = builder.Build();

var chat = kernel.GetRequiredService<IChatCompletionService>();
var history = new ChatHistory();

// Get the reference the document, convert it to embeddings, and store it in ChromaDB.
using var reader = new StreamReader("C:\\Elastacloud\\Training\\RetrievalAugmentedGeneration\\Data\\background-data.txt");
var textContents = await reader.ReadToEndAsync();

var chunksToEmbed = TextChunker.SplitPlainTextLines(textContents, maxTokensPerLine: 40);
var embeddings = await embeddingService.GenerateEmbeddingsAsync(chunksToEmbed, kernel);

if (!await store.DoesCollectionExistAsync("embeddings"))
{
    await store.DeleteCollectionAsync("embeddings");
    await store.CreateCollectionAsync("embeddings");
}

var records = chunksToEmbed.Zip(embeddings).Select(((string text, ReadOnlyMemory<float> embedding) tuple, int idx) => MR.LocalRecord(
    idx.ToString(),
    text: tuple.text,
    description: "Information about Star Trek Deep Space Nine",
    embedding: tuple.embedding,
    timestamp: DateTime.Now
)).ToList();

await foreach (var res in store.UpsertBatchAsync("embeddings", records))
{
    Console.WriteLine(res);
}

history.AddSystemMessage(@"
    You are an assistant that reads from a children's book called The Tale of Don Ni and Natali,
    and help people by answering questions from it.
    You always try to adhere as strictly as possible to the information in that book.
    The document is about a fictional character named Don Ni, and all questions will be about it."
);

await ollamaChat.InitChat(history);

var exit = false;
while (!exit)
{
    var userMessage = Console.ReadLine() ?? "";
    
    var correspondingEmbeddings = await embeddingService.GenerateEmbeddingAsync(userMessage, kernel);

    await foreach(var match in store.GetNearestMatchesAsync("embeddings", correspondingEmbeddings, limit: 5))
    {
        var text = match.Item1.Metadata.Text;
        Console.WriteLine($"INFO: {text}");
        history.AddSystemMessage($"Here's what you know about Don Ni regarding the question that was asked: {text}");
    }

    history.AddMessage(AuthorRole.User, userMessage);
    Console.Write("\n\nResponse: ");
    await foreach (var character in chat.GetStreamingChatMessageContentsAsync(history))
    {
        Console.Write(character);
    };

    Console.WriteLine("\n\nDo you wish to exit? [Y/N]");
    exit = (Console.ReadLine() ?? "N").Equals("Y", StringComparison.CurrentCultureIgnoreCase);
}