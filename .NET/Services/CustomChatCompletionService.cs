using System.Runtime.CompilerServices;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using OllamaSharp;
using OllamaSharp.Models.Chat;

namespace RetrievalAugmentedGeneration.Services;

class CustomChatCompletionService: IChatCompletionService
{
    public readonly OllamaApiClient _ollamaApiClient;
    public readonly Chat _chat;

    public CustomChatCompletionService(string modelUrl, string modelName)
    {
        _ollamaApiClient = new OllamaApiClient(modelUrl, modelName);
        _chat = new Chat(_ollamaApiClient);
    }
    
    public IReadOnlyDictionary<string, object?> Attributes => throw new NotImplementedException();

    public async Task InitChat(ChatHistory chatHistory, CancellationToken cancellationToken = default)
    {
        var models = (await _ollamaApiClient.ListLocalModels(cancellationToken)).ToList();

        if (models.Count == 0)
        {
            var idx = 0;
            await foreach (var status in _ollamaApiClient.PullModel("llama3.2:1b", cancellationToken: cancellationToken))
            {
                if (idx % 100 == 0) Console.WriteLine($"{status!.Percent}% {status.Status}");
                idx += 1;
            }

            _ollamaApiClient.SelectedModel = "llama3.2:1b";
        }

        foreach (var message in chatHistory)
        {
            if (message.Role == AuthorRole.System)
            {
                await _chat.SendAs(ChatRole.System, message.Content!, cancellationToken).StreamToEnd();
                continue;
            }
        }
    }


    public async Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(ChatHistory chatHistory, PromptExecutionSettings? executionSettings = null, Kernel? kernel = null, CancellationToken cancellationToken = default)
    {
        foreach (var message in chatHistory)
        {
            if (message.Role == AuthorRole.System)
            {
                await _chat.SendAs(ChatRole.System, message.Content!, cancellationToken).StreamToEnd();
                continue;
            }
        }

        var lastMessage = chatHistory.LastOrDefault();

        string question = lastMessage!.Content!;
        var chatResponse = "";
        var history = await _chat.Send(question, CancellationToken.None).StreamToEnd();
        chatResponse = history;

        chatHistory.AddAssistantMessage(chatResponse);

        return chatHistory;
        
    }

    public async IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(ChatHistory chatHistory, PromptExecutionSettings? executionSettings = null, Kernel? kernel = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        foreach (var message in chatHistory)
        {
            if (message.Role == AuthorRole.System)
            {
                await _chat.SendAs(ChatRole.System, message.Content!, cancellationToken).StreamToEnd();
                continue;
            }
        }

        var lastMessage = chatHistory.LastOrDefault();

        string question = lastMessage!.Content!;
        await foreach (var character in _chat.Send(question, CancellationToken.None))
        {
            var content = new StreamingChatMessageContent(AuthorRole.Assistant, character);
            yield return content;
        }
    }
}