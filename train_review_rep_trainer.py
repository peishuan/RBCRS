import os
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from models.review_rep_trainer import ReviewRepTrainer, ReconstructionLoss
from batch_loaders.batch_loader import DialogueBatchLoader
import test_params
from utils import create_dir


# When changing loss function during training, nb of epochs before changing
change_at_epoch = 3


# function that freezes parts of the model
def freeze(model):
    for param in model.encoder.sentence_encoder.parameters():
        param.requires_grad = False
    for param in model.encoder.conversation_encoder.parameters():
        param.requires_grad = False


def train(model, batch_loader, baseline, save_path, nb_epochs, patience,
          use_class_weights=True,
          start_with_class_weights=False, cut_dialogues=-1,
          eval_at_beginning=True, max_num_inputs=None, weight_decay=0):
    """
    Train the Review representation model
    :param cut_dialogues:
    :param patience:
    :param nb_epochs:
    :param save_path:
    :param baseline:
    :param batch_loader:
    :param model:
    :param start_with_class_weights: if True, use class weights at the beginning, and remove them after change_at_epoch
     epochs
    :return:
    """
    batch_loader.set_word2id(model.encoder.word2id) 

    epoch = 0
    patience_count = 0
    best_loss = 1e10
    n_train_batches = batch_loader.n_batches["train"]

    training_losses = []
    validation_losses = []
    start_time = time.time()
    criterion = ReconstructionLoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    if eval_at_beginning:
        # Evaluate
        val_loss = model.evaluate(batch_loader=batch_loader, criterion=criterion, subset="valid", batch_input="full")
        print('--------------------------------------------------------------')
        validation_losses.append(val_loss)
        
        # Write logs
        with open(os.path.join(save_path, "logs"), "a+") as f:
            text = "EPOCH {} : losses {} {} TIME {} s \n". \
                format(epoch, -1, val_loss, time.time() - start_time)
            f.write(text)
        # Save model
        is_best = val_loss < best_loss
        best_loss = min(best_loss, val_loss)
        save_checkpoint({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "params": model.params,
            "best_loss": best_loss,
        }, is_best, save_path)

    while epoch < nb_epochs:
        model.train()
        # do not use weights anymore, freeze a part of the model
        # if start_with_class_weights and epoch >= change_at_epoch:
        #     criterion.liked_criterion = nn.NLLLoss()
        #     freeze(model)
        losses = []
        for _ in tqdm(range(n_train_batches)):
            if cut_dialogues == "epoch":
                batch = batch_loader.load_batch(subset="train", cut_dialogues=epoch + 1)
            else:
                batch = batch_loader.load_batch(subset="train", cut_dialogues=cut_dialogues)
            if model.cuda_available:
                batch["reviews"] = batch["reviews"].cuda()
                batch["target"] = batch["target"].cuda()

            # Train iteration: forward, backward and optimize
            optimizer.zero_grad()
            outputs = model(batch)
           
            loss = criterion(outputs, batch["target"].float())
            #print("loss:{}".format(loss.size()))
            # optimize
            
            #修改將輸出變為一個scalar, RuntimeError: grad can be implicitly created only for scalar outputs
            loss.backward(loss.clone().detach())
            optimizer.step()
            loss = loss.detach()

            # keep losses in memory
            losses.append(loss)
            #epoch_loss = criterion.normalize_loss_reset(np.sum(losses))
            torch.cuda.empty_cache()
        print('Epoch : {} Training Loss : {}'.format(epoch, torch.mean(torch.stack(losses))))
        training_losses.append(torch.mean(torch.stack(losses)))
        
        
        # Evaluate
        val_loss = model.evaluate(batch_loader=batch_loader, criterion=criterion, subset="valid", batch_input="full")

        print('--------------------------------------------------------------')
        validation_losses.append(val_loss)
        epoch += 1

        with open(os.path.join(save_path, "logs"), "a+") as f:
            text = "EPOCH {} : losses {} {} TIME {} s \n". \
                format(epoch, training_losses[-1], val_loss, time.time() - start_time)
            f.write(text)
        # Keep track of best loss for early stopping (disabled if before the loss change)
        if not start_with_class_weights or epoch >= change_at_epoch:
            is_best = val_loss < best_loss
            best_loss = min(best_loss, val_loss)
        else:
            # if start with class weights == True and epoch < change_at_epoch, do not update best_loss.
            is_best = True
        save_checkpoint({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "params": model.params,
            "best_loss": best_loss,
        }, is_best, save_path)

        # Early stopping
        if is_best:
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print("Early stopping, {} epochs without best".format(patience_count))
                break

    print("Training done.")


def evaluate(model, batch_loader, resume, baseline, subset="valid", show_wrong=False):
    # set word2id in batchloader from encoder
    if baseline:
        batch_loader.set_word2id(model.gensen.task_word2id)
    else:
        batch_loader.set_word2id(model.encoder.word2id)
    if not os.path.isfile(resume):
        raise ValueError("no checkpoint found at '{}'".format(resume))

    print("=> loading checkpoint '{}'".format(resume))
    if model.cuda_available:
        checkpoint = torch.load(resume)
    else:
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {} with loss {})"
          .format(resume, checkpoint['epoch'], checkpoint['best_loss']))

    model.evaluate(batch_loader, print_matrices=True, subset=subset, show_wrong=show_wrong)


def save_checkpoint(state, is_best, path):
    torch.save(state, os.path.join(path, "checkpoint"))
    if is_best:
        shutil.copy(os.path.join(path, "checkpoint"), os.path.join(path, "model_best"))


def explore_params(params_seq, baseline=False, data="standard"):
    """
    :param params_seq: sequence of tuples (save_folder, model_params, train_params)
    :return:
    """
    #設定model
    model_class = ReviewRepTrainer
    sources = "review_rep"

    #參數設定 儲存路徑 MODEL設定 TRAINING設定
    for (save_path, params, train_params) in params_seq:
        #建立儲存model資料夾
        create_dir(save_path)
        print("Saving in {} with parameters : {}, {}".format(save_path, params, train_params))

        # start_with_class_weights': False => whether to use class weights to reduce class imbalance for liked? label
        if "start_with_class_weights" in train_params and train_params["start_with_class_weights"]:
            train_params["use_class_weights"] = True
            print("start_with_class_weights is set to True, setting use_class_weights=True")
        else:
            train_params["start_with_class_weights"] = False
        
        # use_class_weights => [1. / 5, 1. / 80, 1. / 15]
        if train_params["use_class_weights"]:
            train_params["use_class_weights"] = [1. / 5, 1. / 80, 1. / 15]

        #data = standard
        if data == "standard":
            batch_loader = DialogueBatchLoader(
                sources=sources, # "sentiment_analysis movie_occurrences"
                batch_size=train_params['batch_size'] # 1
            )
            sa = model_class(train_vocab=batch_loader.train_vocabulary,params=params,n_movies=batch_loader.n_movies)
            
            train(
                sa,
                nb_epochs=train_params["nb_epochs"],
                patience=train_params["patience"],
                save_path=save_path,
                baseline=baseline,
                batch_loader=batch_loader,
                use_class_weights=train_params['use_class_weights'],
                start_with_class_weights=train_params["start_with_class_weights"],
                cut_dialogues=train_params['cut_dialogues'] #-1; don't cut dialogues
            )

if __name__ == '__main__':
    #載入參數 學習review representation
    params_seq = [(config.REVIEW_REP_MODEL, test_params.review_rep_params, test_params.train_review_rep_params)]
    #設定初始參數及載入資料
    explore_params(params_seq, baseline=False)
