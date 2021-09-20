# Model hyper parameters
decoder_params = {
    "hidden_size": 256,
    "num_layers": 1,
    "peephole": False
}

hrnn_params = {
    'use_movie_occurrences': False,
    'sentence_encoder_hidden_size': 256,
    'conversation_encoder_hidden_size': 256,
    'sentence_encoder_num_layers': 1,
    'conversation_encoder_num_layers': 1,
    'use_dropout': False,
}

hred_params = {
    'decoder_params': decoder_params,
    "hrnn_params": hrnn_params
}
sentiment_analysis_baseline_params = {
    'use_dropout': False,
    'hidden_size': 512,
    'sentence_encoder_num_layers': 2
}
sentiment_analysis_params = {
    'hrnn_params': {
        # whether to add a dimension indicating the occurrence of the movie name in the sentence
        'use_movie_occurrences': 'word',
        'sentence_encoder_hidden_size': 512,
        'conversation_encoder_hidden_size': 512,
        'sentence_encoder_num_layers': 2,
        'conversation_encoder_num_layers': 2,
        'use_dropout': 0.4,
    }
}
review_rep_params = {
    'layer_sizes': [1000],
    'f': "sigmoid",
    'g': "sigmoid",
    'use_movie_occurrences': 'word',
    'sentence_encoder_hidden_size': 512,
    'conversation_encoder_hidden_size': 512,
    'sentence_encoder_num_layers': 2,
    'conversation_encoder_num_layers': 2,
    'use_dropout': 0.4,
    
}
review_weighted_rec_params = {
    'conversation_encoder_hidden_size': 512,
    "sentiment_analysis_params": sentiment_analysis_params
}
autorec_params = {
    'layer_sizes': [1000],
    'f': "sigmoid",
    'g': "sigmoid",
}
recommend_from_dialogue_params = {
    "sentiment_analysis_params": sentiment_analysis_params,
    "autorec_params": autorec_params
}
enhanced_recommend_from_dialogue_params = {
    "sentiment_analysis_params": sentiment_analysis_params,
    "rwr_params": review_weighted_rec_params
}
recommender_params = {
    'decoder_params': decoder_params,
    'hrnn_params': hrnn_params,
    'recommend_from_dialogue_params': recommend_from_dialogue_params,
    'latent_layer_sizes': None,
    'language_aware_recommender': False,
}
enhanced_recommender_params = {
    'decoder_params': decoder_params,
    'hrnn_params': hrnn_params,
    'enhanced_recommend_from_dialogue_params': enhanced_recommend_from_dialogue_params,
    'latent_layer_sizes': None,
    'language_aware_recommender': False,
}

# Training parameters
train_sa_params = {
    "learning_rate": 0.001,
    "batch_size": 16,
    "nb_epochs": 50,
    "patience": 5,

    "weight_decay": 0,
    "use_class_weights": True,  # whether to use class weights to reduce class imbalance for liked? label
    "cut_dialogues": -1,  # if >=0, specifies the width of the cut around movie mentions. Otherwise, don't cut dialogues
    "targets": "suggested seen liked"
}

train_review_rep_params = {
    "learning_rate": 0.001,
    "batch_size": 1,
    "nb_epochs": 50,
    "patience": 5,

    "weight_decay": 0,
    "use_class_weights": False,  # whether to use class weights to reduce class imbalance for liked? label
    "cut_dialogues": -1,  # if >=0, specifies the width of the cut around movie mentions. Otherwise, don't cut dialogues
    "targets": "suggested seen liked"
}

train_review_weighted_rec_params = {
    "learning_rate": 0.001,
    "batch_size":32,
    "nb_epochs": 50,
    "patience": 5,

    "use_class_weights": False,  # whether to use class weights to reduce class imbalance for liked? label
    "review_weighted_rec_params": review_weighted_rec_params
}

train_autorec_params = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "nb_epochs": 50,
    "patience": 5,

    "batch_input": "random_noise",
    "max_num_inputs": 1e10  # max number of inputs (for random_noise batch loading mode)
}
train_recommender_params = {
    "learning_rate": 0.001,
    "batch_size": 1,
    "nb_epochs": 50,
    "patience": 5,
}
train_enhanced_recommender_params = {
    "learning_rate": 0.001,
    "batch_size": 1,
    "nb_epochs": 50,
    "patience": 5,
}
