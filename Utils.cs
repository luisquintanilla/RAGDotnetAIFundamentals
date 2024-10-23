public class Utils
{
    public static List<Movie> LoadMovies()
    {
        var movieData = new List<Movie>()
        {
            new Movie
                {
                    Title="Lion King", 
                    Description="The Lion King is a classic Disney animated film that tells the story of a young lion named Simba who embarks on a journey to reclaim his throne as the king of the Pride Lands after the tragic death of his father."
                },
            new Movie
                {
                    Title="Inception", 
                    Description="Inception is a science fiction film directed by Christopher Nolan that follows a group of thieves who enter the dreams of their targets to steal information."
                },
            new Movie
                {
                    Title="The Matrix", 
                    Description="The Matrix is a science fiction film directed by the Wachowskis that follows a computer hacker named Neo who discovers that the world he lives in is a simulated reality created by machines."
                },
            new Movie
                {
                    Title="Shrek", 
                    Description="Shrek is an animated film that tells the story of an ogre named Shrek who embarks on a quest to rescue Princess Fiona from a dragon and bring her back to the kingdom of Duloc."
                }
        };

        return movieData;        
    }
}