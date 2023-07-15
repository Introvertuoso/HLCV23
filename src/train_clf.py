
import torch
import numpy as np






def get_acc(gt, preds = None):
    if preds is not None: 
        return ((preds.argmax(1)==gt).sum()/len(preds)).cpu().numpy()
        
    
    return ((preds.argmax(1)==gt).sum()/len(preds)).cpu().numpy()
    

def evaluate(emb_model, model, val_loader, loss_fn, feature_fn, device='cuda'):
    eval_acc = []
    eval_losses = []
    for eval_batch in val_loader:
        ims, labels = eval_batch
        ims, labels = ims.to(device), labels.to(device)
        with torch.no_grad():
            features = feature_fn(emb_model, ims).squeeze()
            preds = model(features)
            loss_val = loss_fn(preds, labels.view(-1,))
            val_acc = get_acc(labels.view(-1,), preds)
        
        eval_acc.append(val_acc)
        eval_losses.append(loss_val.item())
    
    return np.mean(eval_losses), np.mean(eval_acc)
            
            
            


def train(emb_model, clf_model, optim, loss_fn, train_loader, val_loader, feature_fn, epochs=30, device='cuda'):
    losses = []
    accs = []
    val_losses = []
    val_accs = []
    for ep in range(epochs):
        run_loss = 0.
        ep_losses = []
        ep_accs = []

        
        eval_loss, eval_acc = evaluate(emb_model=emb_model, model=clf_model, 
                                       val_loader=val_loader, loss_fn=loss_fn, 
                                       feature_fn=feature_fn, device=device) 
        if ep==0:
            print(f'initial loss {eval_loss} and initial accuracy {eval_acc}')
        
        for i, batch in enumerate(train_loader, 0):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            optim.zero_grad()
            features = feature_fn(emb_model, imgs).squeeze()
            preds = clf_model(features.float())
            # print(preds.argmax(1), labels.view(-1, ))
            loss = loss_fn(preds, labels.view(-1, ))
                
            loss.backward()
            optim.step()
            
            ep_losses.append(loss.item())
            ep_accs.append(get_acc(labels.view(-1,), preds))
            
        ep_loss = np.mean(ep_losses)
        losses.append(ep_loss)
        
        ep_acc = np.mean(ep_accs)
        accs.append(ep_acc)
        
        eval_loss, eval_acc = evaluate(emb_model=emb_model, model=clf_model, 
                                       val_loader=val_loader, loss_fn=loss_fn, 
                                       feature_fn=feature_fn, device=device) 
        val_losses.append(eval_loss)
        val_accs.append(eval_acc)
        print(f' train loss: {ep_loss}, val loss: {eval_loss}, Train accuracy {ep_acc}, val accuracy {eval_acc} ')
        

    return losses, accs, val_losses, val_accs