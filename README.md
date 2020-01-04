# DQN Pong


| ![](gifs/neg_8.gif) | ![](gifs/openai_gym.gif) | ![](gifs/good_start.gif) |


## OpenAI Pong-V0 Gym Environment

- First 34 rows is score, white bounding bar - cut out for training, execution
- Last 16 rows is white bounding bar - cut this out
- Action Space: integers [0,1,2,3,4,5] - represents movements of [do nothing, do nothing, up, down, up, down]
- Background colour: (144  72  17)
- Unique Colours:
```
         RGB             Count      Description
         [  0   0   0]   8          Black 
         [ 92 186  92]   256        Our Paddle
         [144  72  17]   28984      Background
         [213 130  74]   192        Opponent Paddle
         [236 236 236]   4160       White Boundary
```
- Paddle Positions: [start,end] inclusive
```
         Opponent        Us         Ball
         [16,20]         [140,144]  [20,140]
```

## Poster Feedback
- Limit bias where possible (Remove better reward and be careful with prefill memory)
- Try LSTM/GRU with state space of [our_y, opp_y, ball_x, ball_y]
- Reduce size of memory buffer. Try 20,000 - 50,000. If too large, it'll keep seeing too much old memory and not learn effectively
- Compare to conv-net
- Take research approach. Justify why our method is ideal
- Should be able to get a score > 0