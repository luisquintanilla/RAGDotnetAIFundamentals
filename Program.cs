#pragma warning disable
using System.Collections.ObjectModel;
using System.Linq;
using Microsoft.Extensions.VectorData;
using Microsoft.ML.Tokenizers;
using Microsoft.SemanticKernel.Connectors.InMemory;

// Initialize tokenizer and model
var tokenizer = BertTokenizer.Create(Path.Join("assets", "vocab.txt"));

// Create embedding generator
var generator = new MLNETOnnxEmbeddingGenerator(tokenizer, Path.Join("assets","model.onnx"));

// Initialize movie service
var movieService = new MovieService(new InMemoryVectorStore(), generator);

// Populate VectorStore
await movieService.LoadAsync();

var query = "A family friendly movie";

// Search for movies similar to the query
var searchOptions = new VectorSearchOptions()
{
    Top = 1,
    VectorPropertyName = "Vector"
};

var results = await movieService.SearchAsync(query, searchOptions);

// Display search results
await foreach(var result in results.Results)
{
    Console.WriteLine($"Title: {result.Record.Details.Title}");
    Console.WriteLine($"Description: {result.Record.Details.Description}");
    Console.WriteLine($"Score: {result.Score}");
    Console.WriteLine();
}