"""
Modified from
        https://dspace.mit.edu/bitstream/handle/1721.1/3265/P-2108-26912652.pdf;sequence=1
"""
import torch

def auction_lap(X, eps=None, compute_score=True):
    """
        X: n-by-n matrix w/ integer entries
        eps: "bid size" -- smaller values means higher accuracy w/ longer runtime
    """
    eps = 1 / X.shape[0] if eps is None else eps
    
    # --
    # Init
    
    cost     = torch.zeros((1, X.shape[1]))
    curr_ass = torch.zeros(X.shape[0]).long() - 1
    bids     = torch.zeros(X.shape)
    
    if X.is_cuda:
        cost, curr_ass, bids = cost.cuda(), curr_ass.cuda(), bids.cuda()
    
    counter = 0
    while (curr_ass == -1).any():
        counter += 1
        
        # --
        # Bidding
        
        unassigned = torch.nonzero(curr_ass == -1).squeeze()
        if len(unassigned.shape) < 1:
            unassigned = unassigned.unsqueeze(0)
        value = X[unassigned] - cost
        top_value, top_idx = value.topk(2, dim=1)
        
        first_idx = top_idx[:,0]
        first_value, second_value = top_value[:,0], top_value[:,1]
        
        bid_increments = first_value - second_value + eps
        
        bids_ = bids[unassigned]
        bids_.zero_()
        if len(bids_.shape) < 2:
            bids_ = bids_.unsqueeze(dim=0)
        bids_.scatter_(
            dim=1,
            index=first_idx.contiguous().view(-1, 1),
            src=bid_increments.view(-1, 1)
        )
        
        # --
        # Assignment
        
        have_bidder = torch.nonzero((bids_ > 0).int().sum(dim=0))
        
        high_bids, high_bidders = bids_[:,have_bidder].max(dim=0)
        high_bidders = unassigned[high_bidders.squeeze()]
        
        cost[:,have_bidder] += high_bids
        
        curr_ass[(curr_ass.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1)] = -1
        curr_ass[high_bidders] = have_bidder.squeeze()
        if counter > 1000:
            curr_ass = torch.clamp(curr_ass, min=0)
    score = None
    if compute_score:
        score = int(X.gather(dim=1, index=curr_ass.view(-1, 1)).sum())
    
    return score, curr_ass, counter