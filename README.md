# DQN Pong

![](assets/carb.jpg)

## OpenAI Pong-V0 Gym Environment

- First 34 rows is score, white bounding bar - cut out for training, execution
- Last 16 rows is white bounding bar - cut this out
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
