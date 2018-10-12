## Alexa, open the pod bay doors.

![Alexa](images/alexa.2.jpg)

##### An analysis of Echo Dot reviews using Natural Language Processing
#### Data Source

The data I drew my analysis from was ~13,000 reviews of Amazon's Echo Dot (2nd Generation).  There were four columns in the file:
    - Date of Review (dd/mm/yy)
    - Review Header (string)
    - Review Text (string)
    - Review Rating (integer)

The reviews are mostly positive.  Here are the distribution of the star ratings:

![Histogram](images/Histogram_of_Stars.png)

Example of a 5-star review taken from Amazon:

![5_Star_Review](images/5Star_Echo_Review.png)

Example of a 1-star review taken from Amazon:

![1_Star_Review](images/1Star_Echo_Review.png)


#### Natural Language Processing

###### Bag-of-Words Model

The Bag-of-Words Model is a simplifying representation of large amounts of text used in Natural Language Processing.  It is also known as the vector space model. It transforms all the words in a set of documents into a single list of unique words and then creates a matrix, assigning each unique word to a column, and each document (in this case each review) to an individual row. As you can see in a simplified version below, each field is then assigned a numerical value based on the number of occurrences of each word in that particular document.

Creating a list of unique words:

![vocab_demonstration](images/vocab_demo.png)

Assigning values to each field:

![vector_demonstration](images/vector_demo.png)

######   N-grams

N-grams are a continues sequence of n number of words.  In certain cases, it is better to analyze text using sequences of words in order to derive a more complete context of meaning.  

Here is an example of a bi-gram:

![ngram_demonstration](images/ngram_demo.png)

###### TFIDF

TFIDF stands for Term Frequency-Inverse Document Frequency.  This is a method for weighting different words based on their frequency within a given document and their frequency within the entire set of documents.  The idea behind this is that words appearing across a large number of documents are less significant.

#### Cleaning

In order to prepare the reviews to be tranformed into a TFIDF matrix, a certain amount of cleaning needed to be done to avoid unnecessary word duplicates (i.e. alexa, Alexa, and alexa!)

    - Remove punctuation
    - Remove stopwords (words that appear frequently but have no meaning)
    - Remove non-alphabetic characters
    - Strip extra spaces
    - Change all words to lowercase



![elbow_plot_comparison](images/elbow_plot_comparison.png)

#### Results

|    | 4-gram_5_star_review           | 4-gram_1_star_review            |
|---:|:-------------------------------|:--------------------------------|
|  0 | add things shopping list       | not stay connected wifi         |
|  1 | add something shopping list    | play music already downloaded   |
|  2 | add items shopping list        | not worth money hope            |
|  3 | play music throughout house    | would not recommend anyone      |
|  4 | one upstairs one downstairs    | would not recommend product     |
|  5 | one living one bedroom         | half time doesn answer          |
|  6 | easy set easy use              | play amazon prime music         |
|  7 | makes life much easier         | could get money back            |
|  8 | adding items shopping list     | every time ask something        |
|  9 | play music answer questions    | would not stay connected        |
| 10 | didn think would like          | asking questions not really     |
| 11 | thinking getting another one   | almost completely stopped using |
| 12 | would highly recommend product | get questions answered not      |
| 13 | tp link smart plug             | would not connect wifi          |
| 14 | kids love asking questions     | not stay connected internet     |

![key_words](images/key_words_2.gif)
### Future Work
### References
