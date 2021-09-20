import os

path = os.path.dirname(os.path.basename(__file__))

# models path
MODELS_PATH = './models/'
AUTOREC_MODEL = os.path.join(MODELS_PATH, "autorec/seq_train/")
SENTIMENT_ANALYSIS_MODEL = os.path.join(MODELS_PATH, 'sentiment_analysis')
REVIEW_REP_MODEL = os.path.join(MODELS_PATH, 'review_I_A_rep/unifying_sampling_like_review5')
REVIEW_WEIGHTED_REC_MODEL = os.path.join(MODELS_PATH, 'review_weighted_rec/final/seq/pretrain')
ENHANCED_RECOMMENDER_MODEL = os.path.join(MODELS_PATH, "enhanced_recommender/final/experi/seq_pretrain")
RECOMMENDER_MODEL = os.path.join(MODELS_PATH, "recommender/rating_seq")
# data set path
REDIAL_DATA_PATH = './redial/'

TRAIN_PATH = "train_data"
VALID_PATH = "valid_data"
TEST_PATH = "test_data"


REVIEW_DATA_PATH = './external_knowledge_base/like_movie_review/unifying_sampling_review5/'
REDIAL_REVIEW_KNOWLEDGE_TRAIN= "train_like_review5"
REDIAL_REVIEW_KNOWLEDGE_TEST= "test_like_review5"
REDIAL_REVIEW_KNOWLEDGE_VALID= "valid_like_review5"

RWR_DATA_PATH = './external_knowledge_base/review_weighted_rec_training/experi/seq/'
RWR_TRAIN= "train"
RWR_TEST= "test"
RWR_VALID= "valid"

REVIEW_PATH = './external_knowledge_base/review_knowledge_base.csv'
#matched_review_knowledge_id_text.csv'
MOVIE_REP_MATRIX_PATH = './review_enhanced_movie_rep/expanded_movie_rep_I_A_like.pkl'
MOVIE_PATH = os.path.join(REDIAL_DATA_PATH, "movies_merged.csv")
VOCAB_PATH = os.path.join(REDIAL_DATA_PATH, "vocabulary.p")

# reddit data path
# (If you want to pre-train the model on the movie subreddit, from the FB movie dialog dataset)
# Note: this was not used to produce the results in "Towards Deep Conversational Recommendations" as it did not
# produce good results for us.
# REDDIT_PATH = "/path/to/fb_movie_dialog_dataset/task4_reddit"
# REDDIT_TRAIN_PATH = "task4_reddit_train.txt"
# REDDIT_VALID_PATH = "task4_reddit_dev.txt"
# REDDIT_TEST_PATH = "task4_reddit_test.txt"

REVIEW_LENGTH_LIMIT = 10  # reviews are truncated after 10 sentences
REVIEW_SENTENCE_WORD_LENGTH_LIMIT = 50  # sentences are truncated after 20 words

CONVERSATION_LENGTH_LIMIT = 40  # conversations are truncated after 40 utterances
UTTERANCE_LENGTH_LIMIT = 80  # utterances are truncated after 80 words

# Movielens ratings path
ML_DATA_PATH = "./movielens/"
ML_SPLIT_PATHS = [os.path.join(ML_DATA_PATH, "split0"),
                  os.path.join(ML_DATA_PATH, "split1"),
                  os.path.join(ML_DATA_PATH, "split2"),
                  os.path.join(ML_DATA_PATH, "split3"),
                  os.path.join(ML_DATA_PATH, "split4"), ]
ML_TRAIN_PATH = "train_ratings"
ML_VALID_PATH = "valid_ratings"
ML_TEST_PATH = "test_ratings"