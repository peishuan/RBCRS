from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import csv
import re
from models.recommender_model import Recommender as Ori_rec
from models.enhanced_recommender_model import Recommender as Enh_rec
from sequence_generator import SequenceGenerator
from batch_loaders.batch_loader import DialogueBatchLoader
from utils import load_model
from beam_search import get_best_beam
import test_params

# python generate_responses_intersection.py --model_path_ori=./models/recommender/BCE_lr0001bs128_moviefeedback/model_best --model_path_enh=./models/enhanced_recommender/review_only_samerand_IA_1to1_lr0001bs128_moviefeedback/model_best --save_path=./generated_text/int_train_moviefeedback.txt 
def get_movies(path):
    # 載入電影資料
    id2name = {}
    db2id = {}
    
    with open(path, 'r',encoding="utf-8") as f:
    #開啟檔案
        reader = csv.reader(f)
        # 以csv reader讀取
        date_pattern = re.compile(r'\(\d{4}\)')
        # remove date from movie name 去除電影年分

        for row in reader:
            if row[0] != "index":
            #如果第一行非index
                id2name[int(row[0])] = date_pattern.sub('', row[1])
                # id2name [index] = 電影名稱
                db2id[int(row[2])] = int(row[0])
                # db2id[databaseId] = index; db2id is a dictionary mapping ReDial movie Ids to global movieId
    del db2id[-1]
    print("loaded {} movies from {}".format(len(id2name), path))
    return id2name, db2id

def ori_rec_generated():
    with open(args.save_path, "w") as f:
        f.write("")
    #計算推薦比例
    do_rec = False
    do_rec_list = []

    #計算推薦次數
    rec_counter = 0
    rec_times_list = []

    #計算成功比例
    rec_suss = False
    rec_suss_list = []

    #計算到第幾次成功
    suaa_rec_at = 0
    suss_rec_at_list = []
    suss_rec_total_list = []

    #進行推薦句次
    rec_utt_at = []

    #推薦成功句次
    rec_utt_suss = 0 
    rec_utt_suss_at = []

    #推薦時句次
    rec_total_utt = []

    example_amount = args.n_examples
    rec_index = []
    for _ in tqdm(range(args.n_examples)):
        # Load batch
        
        batch_index = batch_loader.batch_index[args.subset]
        batch = batch_loader.load_batch(subset=args.subset)

        # #取的rec提seeker喜歡list
        id2name, db2id = get_movies(movie_path)
        # #global電影名稱對應id2name
        db2name = {db: id2name[id] for db, id in db2id.items()}
        # print(batch)
        init_q = batch["seeker"]
        # #存入seeker資料
        resp_q = batch["recommender"]
        # #存入recommender資料
        gen = (key for key in init_q if key in resp_q and not db2name[int(key)].isspace())
        # #取得和db有關電影
        # get seen and liked movie list
        recsuccesslist = []
        for key in gen:
            if ((resp_q[key]["suggested"] == 0) & (init_q[key]["suggested"] == 0) & (resp_q[key]["liked"] == 1) & (init_q[key]["liked"] == 1)):
                recsuccesslist.append(db2id[int(key)])

        if ori_rec.cuda_available:
            batch["dialogue"] = batch["dialogue"].cuda()
            batch["target"] = batch["target"].cuda()
            batch["senders"] = batch["senders"].cuda()

        # 1) Compute the contexts and recommendation vectors
        # encoder result: (conv_length, hidden_size)
        conversation_representations = ori_rec.encoder(batch, return_all=True).squeeze()
        #print("conversation_representations: {}".format(conversation_representations.size()))

        # get movie_recommendations
        movie_recommendations = ori_rec.recommender_module(
            dialogue=batch["dialogue"],
            senders=batch["senders"],
            lengths=batch["lengths"],
            conversation_lengths=batch["conversation_lengths"],
            movie_occurrences=batch["movie_occurrences"],
            recommend_new_movies=True,
        ).squeeze()  # (conv_length, n_movies)
        #print("movie_recommendations:{}, \n,{}".format(movie_recommendations.size(),movie_recommendations))
        conv_length = movie_recommendations.data.shape[0]

        # select contexts after seeker's utterances
        # indices of seeker's utterances(< conv_len)
        idx = Variable(torch.nonzero((batch["senders"].view(-1) == 1).data).squeeze())
        if ori_rec.cuda_available:
            idx = idx.cuda()
        #print("id:{}".format(idx))
        conversation_representations = conversation_representations.index_select(0, idx)
        movie_recommendations = movie_recommendations.index_select(0, idx)
        #print("conversation_representations:{} movie_recommendations:{}".format(conversation_representations.size(),movie_recommendations.size()))
        # if first utterance is recommender, add a 0-context at the beginning
        if batch["senders"].data.cpu()[0][0] == -1:
            tt = torch.cuda.FloatTensor if ori_rec.cuda_available else torch.FloatTensor
            conversation_representations = torch.cat((
                Variable(torch.zeros((1, ori_rec.params["hrnn_params"]["conversation_encoder_hidden_size"]), out=tt())),
                conversation_representations), 0)
            movie_recommendations = torch.cat((Variable(torch.zeros((1, ori_rec.n_movies), out=tt())),
                                               movie_recommendations), 0)
            #print("batch00:{} conversation_representations:{} movie_recommendations:{}".format(batch["senders"].data.cpu()[0][0],conversation_representations.size(),movie_recommendations.size()))
            

        # Latent variable
        if ori_rec.params['latent_layer_sizes'] is not None:
            # remember that conversation_representations have been shifted one step in time
            h_prior = conversation_representations
            for layer in ori_rec.prior_hidden_layers:
                h_prior = F.relu(layer(h_prior))
            mu_prior = ori_rec.mu_prior(h_prior)
            logvar_prior = ori_rec.sigma_prior(h_prior)
            # No need of posterior for generation

            # In training, sample from the posterior distribution. At test time, sample from prior.
            mu, logvar = (mu_prior, logvar_prior)
            z = ori_rec.reparametrize(mu, logvar)

            context = torch.cat((conversation_representations, z), 1)
        else:
            context = conversation_representations

        # 2) generate sentences conditioned on the contexts and recommendation vectors
        index = 0
        if args.full_dialogue:
            output_str = "CONVERSATION {} \n".format(batch_index)
        else:
            output_str = ""
        messages = [[batch_loader.id2word[w] for w in sentence[:length]]
                    for (sentence, length) in zip(batch["dialogue"][0].data.cpu().tolist(), batch["lengths"][0])]
        # print(messages)
        #print(batch["dialogue"][0].data.cpu().tolist())

        # keep track of movies mentioned by the model, so that it does not recommend twice the same movie
        mentioned_movies = set()
        
        utterance_time = 0
        rec_times = 1
        first_rec = True
        first_suss = True
        for (i, msg) in enumerate(messages):
            if batch["senders"][0].data[i] == -1:  # sent by recommender: generate response
                # continue
                if args.full_dialogue:
                    output_str += "GROUND TRUTH: " + " ".join(msg) + "\n"
                for temperature in temperatures:
                    # BEAM SEARCH
                    beams = generator_ori.beam_search(
                        [batch_loader.word2id["<s>"]],
                        forbid_movies=mentioned_movies,
                        # add batch dimension
                        context=context[index].unsqueeze(0),
                        movie_recommendations=movie_recommendations[index].unsqueeze(0),
                        sample_movies=True,
                        temperature=temperature
                    )
                    if args.only_best:
                        utterance_time = utterance_time + 1
                        # add best beam
                        best_beam = get_best_beam(beams)
                        if args.full_dialogue:
                            output_str += "GENERATED T={} utt#{}: ".format(temperature, utterance_time)
                        output_str += best_beam.get_string(batch_loader.id2word) + "\n"
                        # update set of mentioned movies
                        mentioned_movies.update(best_beam.mentioned_movies)
                        if list(best_beam.mentioned_movies):
                            do_rec = True
                            if(first_rec):
                                rec_utt_at.append(utterance_time)
                                first_rec=False
                            rec_counter = rec_counter + 1
                            #有做推薦動作
                            if(list(best_beam.mentioned_movies)[0] in recsuccesslist):
                                #推薦成功
                                rec_suss = True
                                if(first_suss):
                                    suaa_rec_at = rec_times
                                    rec_utt_suss = utterance_time
                                    first_suss = False
                                output_str += "----------------------------------------------------------\n"
                                output_str += "recommend success at " + str(rec_times) + "th recommendation.\n"
                                output_str += "recommend success at " + str(utterance_time) + "th utterance.\n"
                                output_str += "----------------------------------------------------------\n"
                                rec_times = rec_times + 1
                                # print(list(best_beam.mentioned_movies)[0])
                            else:
                                rec_times = rec_times + 1
                        #else:
                            #延續對話
                            #utterance_time = utterance_time + 1

                        #print("mentioned movies", mentioned_movies)
                         # show all beams sorted by likelihood
                        # sorted_beams = sorted(beams, key=lambda b: -b.likelihood)
                        # for (beam_rank, beam) in enumerate(sorted_beams):
                        #     if args.full_dialogue:
                        #         output_str += "GENERATED T={}, nb {}: ".format(temperature, beam_rank)
                        #     output_str += beam.get_string(batch_loader.id2word) + "\n"
                    else:
                        # show all beams sorted by likelihood
                        sorted_beams = sorted(beams, key=lambda b: -b.likelihood)
                        for (beam_rank, beam) in enumerate(sorted_beams):
                            if args.full_dialogue:
                                output_str += "GENERATED T={}, nb {}: ".format(temperature, beam_rank)
                            output_str += beam.get_string(batch_loader.id2word) + "\n"
                index += 1
            else:  # sent by seeker
                if args.full_dialogue:
                    output_str += "SEEKER: " + " ".join(msg) + "\n"
        if(do_rec):
            do_rec_list.append(1)
            rec_total_utt.append(utterance_time)
            rec_index.append(batch_index)
            do_rec = False

        if(rec_counter!= 0):
            rec_times_list.append(rec_counter)
            rec_counter = 0

        if(rec_suss):
            rec_suss_list.append(1)
            rec_suss = False
            suss_rec_total_list.append(rec_times)
            suss_rec_at_list.append(suaa_rec_at)
            suaa_rec_at = 0
            rec_utt_suss_at.append(rec_utt_suss)

        output_str += "Total recommendation times: {} \n".format(rec_times-1)
        output_str += "\n"
        with open(args.save_path, "a") as f:
            f.write(output_str)
    with open(args.save_path, "a") as f:
        f.write("% of doing recommendation: " + str(len(do_rec_list)/example_amount))
        if(rec_times_list):
            f.write("\naverage times of doing recommendation: " + str(sum(rec_times_list)/len(rec_times_list)))
        else:
            f.write("\naverage times of doing recommendation: 0")
        f.write("\n% of successful recommendation: " + str(len(rec_suss_list)/len(do_rec_list)))
        if(suss_rec_at_list):
            f.write("\nfirst successful recommendation at " + str(sum(suss_rec_at_list)/len(suss_rec_at_list))+"th time")
        else:
            f.write("\nfirst successful recommendation at -th time")
        
        if(rec_utt_at):
            f.write("\nfirst recommendation at " + str(sum(rec_utt_at)/len(rec_utt_at))+"th utterance")
        else:
            f.write("\nfirst recommendation at -th utterance; no any recommendation")

        if(rec_utt_suss_at):
            f.write("\nfirst successful recommendation at " + str(sum(rec_utt_suss_at)/len(rec_utt_suss_at))+"th utterance")
        else:
            f.write("\nfirst successful recommendation at -th utterance; no any successful recommendation")
        
        if(rec_total_utt):
            f.write("\naverage utterance when doing recommendation: " + str(sum(rec_total_utt)/len(rec_total_utt)))
        else:
            f.write("\naverage utterance when doing recommendation:- , no any recommendation" )
        f.write("\n--------------------------------------------------------------------------------------------------\n")
    return rec_index

def ehnanced_rec_generated(ori_rec_list):
    print("enhanced")
    #with open(args.save_path, "w") as f:
    #    f.write("")
    #計算推薦比例
    enh_do_rec = False
    enh_do_rec_list = []

    #計算推薦次數
    enh_rec_counter = 0
    enh_rec_times_list = []

    #計算成功比例
    enh_rec_suss = False
    enh_rec_suss_list = []

    #計算到第幾次成功
    enh_suaa_rec_at = 0
    enh_suss_rec_at_list = []
    enh_suss_rec_total_list = []

    #進行推薦句次
    enh_rec_utt_at = []

    #推薦成功句次
    enh_rec_utt_suss = 0 
    enh_rec_utt_suss_at = []

    #推薦時句次
    enh_rec_total_utt = []

    example_amount = args.n_examples

    for _ in tqdm(range(args.n_examples)):
        # Load batch
        batch_index = batch_loader.batch_index[args.subset]
        batch = batch_loader.load_batch(subset=args.subset)

        # #取的rec提seeker喜歡list
        id2name, db2id = get_movies(movie_path)
        # #global電影名稱對應id2name
        db2name = {db: id2name[id] for db, id in db2id.items()}
        # print(batch)
        init_q = batch["seeker"]
        # #存入seeker資料
        resp_q = batch["recommender"]
        # #存入recommender資料
        gen = (key for key in init_q if key in resp_q and not db2name[int(key)].isspace())
        # #取得和db有關電影
        # get seen and liked movie list
        recsuccesslist = []
        for key in gen:
            if ((resp_q[key]["suggested"] == 1) & (init_q[key]["liked"] == 1)):
                recsuccesslist.append(db2id[int(key)])

        if enh_rec.cuda_available:
            batch["dialogue"] = batch["dialogue"].cuda()
            batch["target"] = batch["target"].cuda()
            batch["senders"] = batch["senders"].cuda()

        # 1) Compute the contexts and recommendation vectors
        # encoder result: (conv_length, hidden_size)
        conversation_representations = enh_rec.encoder(batch, return_all=True).squeeze()
        #print("conversation_representations: {}".format(conversation_representations.size()))

        # get movie_recommendations
        movie_recommendations = enh_rec.recommender_module(
            dialogue=batch["dialogue"],
            senders=batch["senders"],
            lengths=batch["lengths"],
            conversation_lengths=batch["conversation_lengths"],
            movie_occurrences=batch["movie_occurrences"],
            recommend_new_movies=True,
        ).squeeze()  # (conv_length, n_movies)
        #print("movie_recommendations:{}, \n,{}".format(movie_recommendations.size(),movie_recommendations))
        conv_length = movie_recommendations.data.shape[0]

        # select contexts after seeker's utterances
        # indices of seeker's utterances(< conv_len)
        idx = Variable(torch.nonzero((batch["senders"].view(-1) == 1).data).squeeze())
        if enh_rec.cuda_available:
            idx = idx.cuda()
        #print("id:{}".format(idx))
        conversation_representations = conversation_representations.index_select(0, idx)
        movie_recommendations = movie_recommendations.index_select(0, idx)
        #print("conversation_representations:{} movie_recommendations:{}".format(conversation_representations.size(),movie_recommendations.size()))
        # if first utterance is recommender, add a 0-context at the beginning
        if batch["senders"].data.cpu()[0][0] == -1:
            tt = torch.cuda.FloatTensor if enh_rec.cuda_available else torch.FloatTensor
            conversation_representations = torch.cat((
                Variable(torch.zeros((1, enh_rec.params["hrnn_params"]["conversation_encoder_hidden_size"]), out=tt())),
                conversation_representations), 0)
            movie_recommendations = torch.cat((Variable(torch.zeros((1, enh_rec.n_movies), out=tt())),
                                               movie_recommendations), 0)
            #print("batch00:{} conversation_representations:{} movie_recommendations:{}".format(batch["senders"].data.cpu()[0][0],conversation_representations.size(),movie_recommendations.size()))
            

        # Latent variable
        if enh_rec.params['latent_layer_sizes'] is not None:
            # remember that conversation_representations have been shifted one step in time
            h_prior = conversation_representations
            for layer in enh_rec.prior_hidden_layers:
                h_prior = F.relu(layer(h_prior))
            mu_prior = enh_rec.mu_prior(h_prior)
            logvar_prior = enh_rec.sigma_prior(h_prior)
            # No need of posterior for generation

            # In training, sample from the posterior distribution. At test time, sample from prior.
            mu, logvar = (mu_prior, logvar_prior)
            z = enh_rec.reparametrize(mu, logvar)

            context = torch.cat((conversation_representations, z), 1)
        else:
            context = conversation_representations

        # 2) generate sentences conditioned on the contexts and recommendation vectors

        #選擇ori 有座推薦的conv來看
        
        if batch_index not in ori_rec_list:
            continue
        else:
            index = 0
            if args.full_dialogue:
                output_str = "CONVERSATION {} \n".format(batch_index)
            else:
                output_str = ""
            messages = [[batch_loader.id2word[w] for w in sentence[:length]]
                        for (sentence, length) in zip(batch["dialogue"][0].data.cpu().tolist(), batch["lengths"][0])]
            # print(messages)
            #print(batch["dialogue"][0].data.cpu().tolist())

            # keep track of movies mentioned by the model, so that it does not recommend twice the same movie
            mentioned_movies = set()
        
            enh_utterance_time = 0
            enh_rec_times = 1
            enh_first_rec = True
            enh_first_suss = True
            for (i, msg) in enumerate(messages):
                if batch["senders"][0].data[i] == -1:  # sent by recommender: generate response
                    # continue
                    if args.full_dialogue:
                        output_str += "GROUND TRUTH: " + " ".join(msg) + "\n"
                    for temperature in temperatures:
                        # BEAM SEARCH
                        beams = generator_enh.beam_search(
                            [batch_loader.word2id["<s>"]],
                            forbid_movies=mentioned_movies,
                            # add batch dimension
                            context=context[index].unsqueeze(0),
                            movie_recommendations=movie_recommendations[index].unsqueeze(0),
                            sample_movies=True,
                            temperature=temperature
                        )
                        if args.only_best:
                            enh_utterance_time = enh_utterance_time + 1
                            # add best beam
                            best_beam = get_best_beam(beams)
                            if args.full_dialogue:
                                output_str += "GENERATED T={} utt#{}: ".format(temperature, enh_utterance_time)
                            output_str += best_beam.get_string(batch_loader.id2word) + "\n"
                            # update set of mentioned movies
                            mentioned_movies.update(best_beam.mentioned_movies)
                            if list(best_beam.mentioned_movies):
                                enh_do_rec = True
                                if(enh_first_rec):
                                    enh_rec_utt_at.append(enh_utterance_time)
                                    enh_first_rec=False
                                enh_rec_counter = enh_rec_counter + 1
                                #有做推薦動作
                                if(list(best_beam.mentioned_movies)[0] in recsuccesslist):
                                    #推薦成功
                                    enh_rec_suss = True
                                    if(enh_first_suss):
                                        enh_suaa_rec_at = enh_rec_times
                                        enh_rec_utt_suss = enh_utterance_time
                                        enh_first_suss = False
                                    output_str += "----------------------------------------------------------\n"
                                    output_str += "recommend success at " + str(enh_rec_times) + "th recommendation.\n"
                                    output_str += "recommend success at " + str(enh_utterance_time) + "th utterance.\n"
                                    output_str += "----------------------------------------------------------\n"
                                    enh_rec_times = enh_rec_times + 1
                                    # print(list(best_beam.mentioned_movies)[0])
                                else:
                                    enh_rec_times = enh_rec_times + 1
                            #else:
                                #延續對話
                                #utterance_time = utterance_time + 1

                            #print("mentioned movies", mentioned_movies)
                             # show all beams sorted by likelihood
                            # sorted_beams = sorted(beams, key=lambda b: -b.likelihood)
                            # for (beam_rank, beam) in enumerate(sorted_beams):
                            #     if args.full_dialogue:
                            #         output_str += "GENERATED T={}, nb {}: ".format(temperature, beam_rank)
                            #     output_str += beam.get_string(batch_loader.id2word) + "\n"
                        else:
                            # show all beams sorted by likelihood
                            sorted_beams = sorted(beams, key=lambda b: -b.likelihood)
                            for (beam_rank, beam) in enumerate(sorted_beams):
                                if args.full_dialogue:
                                    output_str += "GENERATED T={}, nb {}: ".format(temperature, beam_rank)
                                output_str += beam.get_string(batch_loader.id2word) + "\n"
                    index += 1
                else:  # sent by seeker
                    if args.full_dialogue:
                        output_str += "SEEKER: " + " ".join(msg) + "\n"
            if(enh_do_rec):
                enh_do_rec_list.append(1)
                enh_rec_total_utt.append(enh_utterance_time)
                enh_do_rec = False

            if(enh_rec_counter!= 0):
                enh_rec_times_list.append(enh_rec_counter)
                enh_rec_counter = 0

            if(enh_rec_suss):
                enh_rec_suss_list.append(1)
                enh_rec_suss = False
                enh_suss_rec_total_list.append(enh_rec_times)
                enh_suss_rec_at_list.append(enh_suaa_rec_at)
                enh_suaa_rec_at = 0
                enh_rec_utt_suss_at.append(enh_rec_utt_suss)

            output_str += "Total recommendation times: {} \n".format(enh_rec_times-1)
            output_str += "\n"
            with open(args.save_path, "a") as f:
                f.write(output_str)
            
        with open(args.save_path, "a") as f:
            f.write("cases of ori doing recommendation: " + str(len(ori_rec_list)))
            if(enh_do_rec_list):
                f.write("\n% of doing recommendation: " + str(len(enh_do_rec_list)/len(ori_rec_list)))
            if(enh_rec_times_list):
                f.write("\naverage times of doing recommendation: " + str(sum(enh_rec_times_list)/len(enh_rec_times_list)))
            else:
                f.write("\naverage times of doing recommendation: 0")
            if(enh_do_rec_list):
                f.write("\n% of successful recommendation: " + str(len(enh_rec_suss_list)/len(enh_do_rec_list)))
            else:
                f.write("\n% of successful recommendation: 0")
            if(enh_suss_rec_at_list):
                f.write("\nfirst successful recommendation at " + str(sum(enh_suss_rec_at_list)/len(enh_suss_rec_at_list))+"th time")
            else:
                f.write("\nfirst successful recommendation at -th time")
        
            if(enh_rec_utt_at):
                f.write("\nfirst recommendation at " + str(sum(enh_rec_utt_at)/len(enh_rec_utt_at))+"th utterance")
            else:
                f.write("\nfirst recommendation at -th utterance; no any recommendation")

            if(enh_rec_utt_suss_at):
                f.write("\nfirst successful recommendation at " + str(sum(enh_rec_utt_suss_at)/len(enh_rec_utt_suss_at))+"th utterance")
            else:
                f.write("\nfirst successful recommendation at -th utterance; no any successful recommendation")
            if(enh_rec_total_utt):
                f.write("\naverage utterance when doing recommendation: " + str(sum(enh_rec_total_utt)/len(enh_rec_total_utt)))
            else:
                f.write("\naverage utterance when doing recommendation:- , no any recommendation" )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path_ori")
    parser.add_argument("--model_path_enh")
    parser.add_argument("--save_path")
    parser.add_argument("--beam_size", default=10)
    parser.add_argument("--n_examples", default=1342)
    parser.add_argument("--only_best", default="True",
                        help="whether to display all the beam results, or only the best")
    parser.add_argument("--full_dialogue", default="True",
                        help="whether to display the full dialogue or only the answers from the model")
    parser.add_argument("--subset", default="test",
                        help="subset on which to condition the model")
    movie_path = './redial/movies_merged.csv'
    args = parser.parse_args()

    temperatures = [1]
    batch_loader = DialogueBatchLoader(
        sources="dialogue movie_occurrences movieIds_in_target",
        batch_size=1
    )
    ori_rec = Ori_rec(
        batch_loader.train_vocabulary,
        batch_loader.n_movies,
        params=test_params.recommender_params
    )
    load_model(ori_rec, args.model_path_ori)
    batch_loader.set_word2id(ori_rec.encoder.word2id)
    generator_ori = SequenceGenerator(
        ori_rec.decoder,
        beam_size=args.beam_size,
        word2id=batch_loader.word2id,
        movie_id2name=batch_loader.id2name,
        max_sequence_length=40
    )
    #do ori rec first
    operating_conv = ori_rec_generated()

    #reload batch data
    batch_loader = DialogueBatchLoader(
        sources="dialogue movie_occurrences movieIds_in_target",
        batch_size=1
    )
    enh_rec = Enh_rec(
        batch_loader.train_vocabulary,
        batch_loader.n_movies,
        batch_loader.movie_rep_matrix,
        params=test_params.enhanced_recommender_params
    )
    load_model(enh_rec, args.model_path_enh)
    batch_loader.set_word2id(enh_rec.encoder.word2id)
    generator_enh = SequenceGenerator(
        enh_rec.decoder,
        beam_size=args.beam_size,
        word2id=batch_loader.word2id,
        movie_id2name=batch_loader.id2name,
        max_sequence_length=40
    )
    #do rnh with intersection
    ehnanced_rec_generated(operating_conv)
    

    
