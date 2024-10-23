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

// Tokenize text
var text = new [] {"The quick brown fox jumps over the lazy dog"};

// Initialize VectorStore Collection
var vectorStore = new InMemoryVectorStore();

var movies = vectorStore.GetCollection<int, MovieEmbedding>("movies");

await movies.CreateCollectionIfNotExistsAsync();

// Generate embeddings and populate VectorStore
var movieData = Utils.LoadMovies();

foreach(var movie in movieData)
{
    var embedding = await generator.GenerateAsync(new [] {movie.Description});
    await movies.UpsertAsync(new MovieEmbedding { 
        Key = movieData.IndexOf(movie),
        Details = movie, 
        Vector = embedding.First().Vector });
}

// Generate embedding for query
var query = "A family friendly movie";
var queryEmbedding = await generator.GenerateAsync(new [] {query});

// Search for movies similar to the query
var searchOptions = new VectorSearchOptions()
{
    Top = 1,
    VectorPropertyName = "Vector"
};

var results = await movies.VectorizedSearchAsync(queryEmbedding.First().Vector, searchOptions);

// Display search results
await foreach(var result in results.Results)
{
    Console.WriteLine($"Title: {result.Record.Details.Title}");
    Console.WriteLine($"Description: {result.Record.Details.Description}");
    Console.WriteLine($"Score: {result.Score}");
    Console.WriteLine();
}