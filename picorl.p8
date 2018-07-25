--- agent ---

function create_agent(x,y)
 local agent={}
 
 agent.x=x
 agent.y=y
 
 return agent
end

function copy_agent(agent)
 return create_agent(agent.x,agent.y)
end

function hash_agent(agent)
 local h=""..agent.x..agent.y
 return h
end

--- cells ---

function create_cell(repr,color)
    local cell={}

    cell.repr=repr
    cell.color=color

    return cell
end

empty=create_cell("e",0)
robot=create_cell("r",11)
wall=create_cell("w",15)
hole=create_cell("h",8)
diamond=create_cell("d",12)

--- grid ---

function create_grid(n)
 local grid={}
 
 grid.n=n
 
 for i=1,n do
  grid[i]={}
  for j=1,n do
   grid[i][j]=empty
  end
 end
 
 return grid
end

function copy_grid(grid)
 local new_grid={}

 new_grid.n=grid.n
 
 for i=1,grid.n do
  new_grid[i]={}
  for j=1,grid.n do
   new_grid[i][j]=grid[i][j]
  end
 end
 
 return new_grid
end

function hash_grid(grid)
    local h=""

    for i=1,grid.n do
     for j=1,grid.n do
      h=h..grid[i][j].repr
     end
    end
    
    return h
   end

function draw_grid(grid)
 local w=flr(128/grid.n)

 for i=1,grid.n do
  for j=1,grid.n do
   local c=grid[i][j].color

   local x0=(i-1)*w
   local y0=(j-1)*w
   local x1=i*w
   local y1=j*w

   rectfill(x0,y0,x1,y1,c)
  end
 end
end

--- state ---

function create_state(grid,agent)
 local state={}
 
 state.grid=grid
 state.agent=agent
 
 return state
end

function copy_state(state)
 local new_agent=copy_agent(state.agent)
 local new_grid=copy_grid(state.grid)
 
 local new_state=create_state(new_grid,new_agent)
 
 return new_state
end

function hash_state(state)
 local h=""
 
 h=h..hash_agent(state.agent)
 h=h..hash_grid(state.grid)

 return h
end

function draw_state(state)
 cls()
 
 -- center camera
 local w=flr(128/state.grid.n)
 local rem=128%w
 camera(-rem/2,-rem/2)
 
 draw_grid(state.grid)
 
 -- draw agent
 local i=state.agent.x
 local j=state.agent.y
 local x0=(i-1)*w
 local y0=(j-1)*w
 local x1=i*w-1
 local y1=j*w-1
 rectfill(x0,y0,x1,y1,robot.color)
 
 -- uncenter camera
 camera(0,0)
end

--[[
    actions:
    up:     1
    down:   2
    left:   3
    right:  4
]]--

function update_state(state,action)
 local done=false
 local reward=0
 
 local n=grid.n
 
 -- agent
 local x0=state.agent.x
 local y0=state.agent.y
 local x1=x0
 local y1=y0

 if action==1 then
  y1=max(1,y1-1)
 elseif action==2 then
  y1=min(n,y1+1)
 elseif action==3 then
  x1=max(1,x1-1)
 elseif action==4 then
  x1=min(n,x1+1)
 end
 
 -- reward and doneness

 local curr=grid[x1][y1].repr
 
 -- this defines the grid+agent behavior
 if curr=="e" then
  reward=-0.01
 elseif curr=="w" then
  reward=-0.02
  x1=x0
  y1=y0
 elseif curr=="h" then
  reward=-1000
  done=true
 elseif curr=="d" then
  reward=10000
  done=true
 end
 
 -- update agent position
 state.agent.x=x1
 state.agent.y=y1
 
 -- return info
 return {reward,done}
end

--- best action tracker ---

function create_tracker(n)
 local tracker={}
 tracker.n=n
 
 for i=1,n do
  tracker[i]={}
  for j=1,n do
   tracker[i][j]=0
  end
 end
 
 return tracker
end

function update_tracker(tracker,learner,state)
 local i=state.agent.x
 local j=state.agent.y
 local opt_action=arg_max_qvalue(learner,state)
 tracker[i][j]=opt_action
end

function draw_tracker(tracker)
 local n=tracker.n
 local w=flr(128/n)
 local rem=128%w
 camera(-rem/2,-rem/2)
 for i=1,n do
  for j=1,n do
   pset((i-1)*w,(j-1)*w,tracker[i][j])
  end
 end
 camera(0,0)
end

--- q-learner ---

function create_qlearner(ep,st,al,ga,rn,n)
 --[[
     ep: episode
     st: max steps/episode
     al: alpha
     ga: gamma
     rn: random action prob
     n: grid size
 ]]--

 local qlearner={}
 
 qlearner.episodes=ep
 qlearner.steps=st
 qlearner.alpha=al
 qlearner.gamma=ga
 qlearner.rnd_action_prob=rn
 qlearner.qtable={}
 
 qlearner.tracker=create_tracker(n)
 
 return qlearner
end

function qvalue(learner,state)
 local h=hash_state(state)
 
 if learner.qtable[h]==nil then
  learner.qtable[h]={0,0,0,0}
 end
 
 return learner.qtable[h]
end

function max_qvalue(learner,state)
 local qval=qvalue(learner,state)
 
 local m=-32767
 for i=1,4 do
  m=max(m,qval[i])
 end
 
 return m
end

function arg_max_qvalue(learner,state)
 local qval=qvalue(learner,state)

 local m=-32767
 local i0=0
 for i=1,4 do
  if m<qval[i] then
   i0=i
   m=qval[i]
  end
 end
 
 return i0
end

function update_qvalue(learner,state,next_state,action,reward)
 local h=hash_state(state)
 local prev_qvalue=qvalue(learner,state)[action]
 local max_next_qvalue=max_qvalue(learner,next_state)
 learner.qtable[h][action] = prev_qvalue + learner.alpha * (reward + learner.gamma * max_next_qvalue - prev_qvalue)
end

function choose_action(learner,state)
 if rnd()<learner.rnd_action_prob then
  return 1+flr(rnd(4))
 else
  return arg_max_qvalue(learner,state)
 end
end

function qlearn(learner,state)
 for e=1,learner.episodes do
  local reward=0
  local curr_state=copy_state(state)

  for s=1,learner.steps do
   -- visual stuff
   draw_state(curr_state)
   draw_tracker(learner.tracker)
   
   color(9)
   print("e:"..e,0,0)
   print("s:"..s,0,6)
   print("1:"..qvalue(learner,curr_state)[1],0,12)
   print("2:"..qvalue(learner,curr_state)[2],0,18)
   print("3:"..qvalue(learner,curr_state)[3],0,24)
   print("4:"..qvalue(learner,curr_state)[4],0,30)
   print("mem:"..stat(0),0,36)

   flip()
   -- end of visual stuff
   
   local next_state=copy_state(curr_state)
   local action=choose_action(learner,curr_state)
   local d=update_state(next_state,action)
   reward+=d[1]
  
   update_qvalue(learner,curr_state,next_state,action,reward)
   update_tracker(learner.tracker,learner,curr_state)
   curr_state=next_state
   
   if d[2] then
    break
   end
  end
 end
end




--- example ---

-- agent: where the agent starts
agent=create_agent(2,2)

-- grid: specify grid/maze cells
grid=create_grid(12)

--[[
-- maze
-- walls
wall_loc={
{2,5},
{3,2},
{3,3},
{3,7},
{3,8},
{3,9},
{3,11},
{4,3},
{4,4},
{4,5},
{4,6},
{4,7},
{5,3},
{5,7},
{5,9},
{5,10},
{5,11},
{6,5},
{6,7},
{7,3},
{7,4},
{7,5},
{7,7},
{7,8},
{7,9},
{7,11},
{8,5},
{9,2},
{9,3},
{9,5},
{9,6},
{9,7},
{9,8},
{9,9},
{9,10},
{9,11},
{10,2},
{10,6},
{11,10}
}

for i=1,12 do
 add(wall_loc,{1,i})
 add(wall_loc,{12,i})
 add(wall_loc,{i,1})
 add(wall_loc,{i,12})
end

n=#wall_loc
for i=1,n do
 x=wall_loc[i][1]
 y=wall_loc[i][2]
 grid[x][y]=wall
end
]]--

-- diagonal obstacles
-- obstacles
for i=2,11 do
 grid[i][i]=hole
end

-- target
grid[12][12]=diamond

-- state: game state (grid+agent)
state=create_state(grid,agent)

-- learner: does q-learning
learner=create_qlearner(32767,2000,0.1,0.95,0.05,12)


-- begin learning
qlearn(learner,state)
