# Agent-based model for COVID-19 transmission in supermarkets. 
This code accompanies the paper ["Modelling COVID-19 transmission in supermarkets using an agent-based model"](https://arxiv.org/abs/2010.07868). This README is based on Fabian Ying's work. Original repository and README can be find [here](https://github.com/fabianying/covid19-supermarket-abm). 

Differences between Fabian Ying's original and ours forked repository:

(1) [Web application (`app.py`)](https://github.com/Saareem/gis-e4030-abm#web-application)

(2) Runtime path generator

(3) Some customers might arrive in pairs

(4) [Staff is included](https://github.com/Saareem/gis-e4030-abm#concept-of-general-agent-and-inclusion-of-staff-members)

(5) In addition of numerical result, also visualizations can be displayed

(6) Popular hours and week days are included

(7) Other minor modifications - like calculating running time - are included



# Installation

Our package relies mainly on packages [covid-19-supermarket_abm](https://pypi.org/project/covid19-supermarket-abm/) and [SimPy](https://simpy.readthedocs.io/en/latest/), which requires Python >= 3.6. Additionally, Flask, Werkzeug and Shapely libraries are used. The code has only been tested on Windows but technically it should also work on other operating systems if the Python version and package version requirements under [`requirements.txt`](https://github.com/Saareem/gis-e4030-abm/blob/main/requirements.txt) can be satisfied.
To get going:

Recommended: Set up virtualenv. Skip if you know what you are doing and know possible consequences.
```bash
# Make sure you have virtualenv installed
> py -m pip install --user virtualenv      # WINDOWS
> python3 -m pip install --user virtualenv # Linux/macOSx

# Create the environment
> py -m venv env                           # WINDOWS
> python3 -m venv env                      # Linux/macOSx

# Activate the environment
> .\env\Scripts\activate                   # WINDOWS
> source env/bin/activate                  # Linux/macOSx

# Now your python and pip should be pointing to those in the env directory

# Deactivate when you are finished
> deactivate
```
Then just install the required packages by:	
```bash
> pip install -r requirements.txt
```
in the root of the gis-e4030-abm

# Example

If you only want to run the application, you can use web application by running `app.py` from the repository root on a local computer. This will start a development server instance on localhost, i.e. [http://127.0.0.1:5000](http://127.0.0.1:5000) by default but the location might vary based on your system configuration. 
 
In the example below, we use locally the example data included in the package to simulate a day in the fictitious store
given the parameters below. 

```python
from covid19_supermarket_abm.utils.load_example_data import load_example_store_graph, load_example_paths
from covid19_supermarket_abm.path_generators import get_path_generator
from covid19_supermarket_abm.simulator import simulate_one_day

# Set parameters
config = {'arrival_rate': 2.55,  # Poisson rate at which customers arrive
           'traversal_time': 0.2,  # mean wait time per node
           'num_hours_open': 14,  # store opening hours
           'infection_proportion': 0.0011,  # proportion of customers that are infectious
         }

# load synthetic data
zone_paths = load_example_paths()
G = load_example_store_graph()

# Create a path generator which feeds our model with customer paths
path_generator_function, path_generator_args = get_path_generator(zone_paths=zone_paths, G=G)

# Simulate a day and store results in results
results_dict = simulate_one_day(config, G, path_generator_function, path_generator_args)
```

The results from our simulations are stored in `results_dict` and/or `result_images`.

```python
print(list(results_dict.keys()))
```
Output:
```python
['num_agents', 'num_S', 'num_I', 'total_exposure_time', 'num_contacts_per_agent', 'num_agents_w_contact', 'mean_num_cust_in_store', 'max_num_cust_in_store', 'num_contacts', 'shopping_times', 'mean_shopping_time', 'num_waiting_people', 'mean_waiting_time', 'store_open_length', 'df_num_encounters_per_node', 'df_exposure_time_per_node', 'total_time_crowded', 'exposure_times', 'logs', 'runtime']
```

See below for their description.

Key | Description
------------ | -------------
`num_agents `| Total number of agents
`num_S` | Number of susceptible agents
`num_I` | Number of infected agents
`total_exposure_time` | Total exposure time
`num_contacts_per_agent` | List of number of contacts with infectious agents per susceptible agent with at least one contact
`num_agents_w_contact` | Number of susceptible agents which have at least one contact with an infectious agent
`mean_num_cust_in_store` | Mean number of customers in the store during the simulation
`max_num_cust_in_store` | Maximum number of customers in the store during the simulation
`num_contacts` | Total number of contacts between infectious agents and susceptible agents
`df_num_encounters_per_node` | Dataframe which contains the the number of encounters with infectious agents for each node
`shopping_times` | Array that contains the length of all customer shopping trips
`mean_shopping_time` | Mean of the shopping times
`num_waiting_people` | Number of people who are queueing outside at every minute of the simulation (when the number of customers in the store is restricted)
`mean_waiting_time` | Mean time that customers wait before being allowed to enter (when the number of customers in the store is restricted)
`store_open_length` | Length of the store's opening hours (in minutes) 
`df_exposure_time_per_node` | Dataframe containing the exposure time per node
`total_time_crowded` | Total time that nodes were crowded (when there are more than `thres` number of agents in a node. Default value of `thres` is 3)
`exposure_times` | List of exposure times of agents (only recording positive exposure times)
`store_open_length` | Length of the store's opening hours in minutes
`runtime` | Total running time per simulated day 

 # Getting started

 As we can see from the above example, our model requires four inputs. 
  ```python
 results_dict = simulate_one_day(config, G, path_generator_function, path_generator_args)
  ```

 These inputs are:

 (1) Simulation configurations: `config`

 (2) A store network: `G`

 (3) A path generator: `path_generator_function`

 (4) Arguments for the path generator: `path_generator_args`

 We discuss each of these inputs in the following subsections.

 ## Simulation configurations

 We input the configuration using a dictionary.
 The following keys are accepted and this might differ a bit in web application:


 ### Mandatory config keys

 Config key | Description
------------ | -------------
`arrival_rate`| Rate at which customers arrive to the store (in customers per minute)
`traversal_time`| Mean wait time at each node (in minutes)
`num_hours_open`| Number of hours that the store is open
`infection_proportion`| Proportion of customers that are infected

 ### Optional config keys

 Config key | Description
------------ | -------------
`max_customers_in_store`| Maximum number of customers allowed in store (Default: `None`, i.e., disabled)
 `with_node_capacity` | Set to `True` to limit the number of customers in each node. (Default: `False`). WARNING: This may cause simulations not to terminate due to gridlocks. 
 `node_capacity` | The number of customers allowed in each node, if  `with_node_capacity` is set to `True`. (Default: `2`)
 `logging_enabled` | Set to `True` to start logging simulations. (Default: `False`). The logs can be accessed in `results_dict['logs']`. Also if sanity checks fail, logs will be saved to file. 
 `duration_days` | The number of days in simulation. If more than `1`, uses simulate_several_days - function. (Default: `1`)
 `day` | Starting week day. Relevant if popular times are defined and user wants to simulate only one day. (Default: `0` i.e. monday)
 `customers_together` | Proportion of customers shopping together. Number between 0 and 1. (Default: `0`)
 `runtime` | Set to `true` to allow customers to avoid each other by using runtime path generators. (Default: `False`) If set to `True`, path generator type also needs to be set accordingly to `runtime`. WARNING: This will make code slower, about 2-4 times with default runtime parameters.
 `staff_start_nodes` | A tuple of the start positions of the staff members in the store graph. For example `(1, 24, 34)` signifies that the staff members will start at nodes 1, 24 and 34 according to the order they are added into the store. The number of elements in the tuple define the number of staff members in the simulation.
 `staff_traversal_time` | Mean wait time at each node (in minutes) for the staff members. If none is supplied, the general `traversal_time` is used. 

In addition, there are optional keys used for runtime path generation. These are detailed in the Runtime path generation -section.
 


 ## Store network

 We use the [NetworkX](https://networkx.org/documentation/stable/) package to create our store network.

 First, we need to specify the (x,y) coordinates of each node. 
 So in a very simple example, we have four nodes, arranged in a square at with coordinates (0,0), (0,1), (1,0), and (1,1).   

 ```python
pos = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
 ```
Next, we need to specify the edges in the network; in other words, which nodes are connected to each other.

 ```python
edges = [(0,1), (1,3), (0,2), (2,3)]
 ```

 We create the graph as follows.
 ```python
from covid19_supermarket_abm.utils.create_store_network import create_store_network
G = create_store_network(pos, edges)
 ```

 To visualize your network, you can use `nx.draw_networkx`:
 ```python
import networkx as nx
nx.draw_networkx(G, pos=pos, node_color='y')
 ```
  
 To create a directed store network network, simply use the `directed=True` parameter in `create_store_network`:
 ```python
from covid19_supermarket_abm.utils.create_store_network import create_store_network
edges = [(0,1), (1,3), (3,1), (0,2), (3,2), (2,3)]
G = create_store_network(pos, edges, directed=True) 
```
 
 ## Popular times per day
 
If you want to include popular times of the real-life market, you can copy the data from Google Maps, for example. 

Create a tsv-file that contains seven columns for each day and multiple rows for every hours. Each cell represent the popularity of specific hour as a values between 1 and 100. The most popular hour is represented as value 100. When the store is closed, the cell is blank. You can see example [here](https://github.com/Saareem/gis-e4030-abm/blob/main/covid19_supermarket_abm/kmarket_data/kmarket_popular_hours.txt).
 
 ## Path generator and arguments

The path generator is what its name suggests: 
It is a [generator](https://wiki.python.org/moin/Generators) that yields full customer paths.

There are three* path generators implemented in this package.

(1) Empirical path generator

(2) Synthetic path generator

(3) Runtime path generator

You can also implement your own path generator and pass it.

To use one of the implemented path generators, 
it is often easiest to use the `get_path_generator` function from the `covid19_supermarket_abm.path_generators` module.

```python
from covid19_supermarket_abm.path_generators import get_path_generator
path_generator_function, path_generator_args = get_path_generator(path_generation, **args) 
```

\*There is a [fourth generator](https://github.com/fabianying/covid19-supermarket-abm/blob/12504eabfad03e2ffe0a6c9aac230d19e24c492a/covid19_supermarket_abm/path_generators.py#L196) implemented, but for most purposes, the first three are likely preferable.

### Empirical path generator 
The empirical path generator takes as input a list of full paths 
(which can be empirical paths or synthetically created paths) and yields random paths from that list.
Note that all paths must be valid paths in the store network or the simulation will fail at runtime.

To use it, simply 
```python
from covid19_supermarket_abm.path_generators import get_path_generator
full_paths = [[0, 1, 3], [0, 2, 3]]  # paths in the store network
path_generator_function, path_generator_args = get_path_generator(path_generation='empirical', full_paths=full_paths) 
```

Alternatively, you can input a list of what we call *zone paths* and the store network `G`.
A zone path is a sequence of nodes that a customer visits, but where consecutive nodes in the sequence need not be adjacent.
In the paper, this sequence represents the item locations of where a customer bought items along with the 
entrance, till and exit node that they visited.
The `get_path_generator` function automatically converts these zone paths to full paths by choosing shortest paths between
consecutive nodes in the zone path.

```python
from covid19_supermarket_abm.path_generators import get_path_generator
zone_paths = [[0, 3], [0, 2, 1], [0, 3, 2]]  # note that consecutive nodes need not be adjacent!
path_generator_function, path_generator_args = get_path_generator(path_generation='empirical', G=G, zone_paths=zone_paths)
```

 ### Synthetic path generator


 The synthetic path generator yields random paths as follows.

 (1) First, it samples the size K of the shopping basket using a [log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution)
  random variable with parameter `mu` and `sigma` 
 (the mean and standard deviation of the underlying normal distribution).
 (See [Sorensen et al, 2017](https://www.sciencedirect.com/science/article/abs/pii/S0969698916303186))

 (2) Second, it chooses a random entrance node as the first node $v_1$ in the path.

 (3) Third, it samples K random item nodes, chosen uniformly at random with replacement from item_nodes, which we denote by
 $v_2, ... v_{K+1}$.

 (4) Fourth, it samples a random till node and exit node, which we denote by $v_{K+2}$ and $v_{K+3}$.
 The sequence $v_1, ..., v_{K+3}$ is a node sequence where the customer bought items, along the the entrance, till and exit
 nodes that they visited.

 (5) Finally, we convert this sequence to a full path on the network using the shortest paths between consecutive nodes
 in the sequence.

 For more information, see the Data section in our [paper](https://arxiv.org/pdf/2010.07868.pdf).

 ```python
from covid19_supermarket_abm.path_generators import get_path_generator
from covid19_supermarket_abm.utils.create_synthetic_baskets import get_all_shortest_path_dicts
import networkx as nx
entrance_nodes = [0]
till_nodes = [2]
exit_nodes = [3]
item_nodes = [1]
mu = 0.07
sigma = 0.76
shortest_path_dict = get_all_shortest_path_dicts(G)
synthetic_path_generator_args = [mu, sigma, entrance_nodes, till_nodes, exit_nodes, item_nodes, shortest_path_dict]
path_generator_function, path_generator_args = get_path_generator(path_generation='synthetic',
                                                            synthetic_path_generator_args=synthetic_path_generator_args)
 ```

 Note that this path generator may be quite slow. In the paper, we first pre-generated paths 100,000 paths 
 and then used the Empirical path generator with the pre-generated paths. 

### Runtime path generator

Runtime path generator creates node sequences using steps 1-4 from synthetic path generation. The sequences aren't converted to full paths using step 5.
During simulation run, the customers update their paths using A* algorithm.
Runtime path generation with default parameters is about 2-4 slower than synthetic path generation, but the performance is dramatically infleunced by the parameters and probably by the network as well.
Runtime path generation uses following parameters given in `config`:

 Runtime parameter key | Description
------------ | -------------
`path_update_interval`| The interval of nodes at which the customers recalculate their paths. Minimum value is 2, but low values cause frequent updating and longer runtimes. (Default: `5`)
`shortest_path_dict` | Not optional, needs to be given. A dictionary of all shortest paths in the graph. Used for calculating the heuristics in A*.
`node_visibility` | Not optional, needs to be given. A dictionary of node pairs, where a value `1`indicates visibility between the nodes, and `0` non-visibility. Used for calculating edge weights in A*. Example on how to calculate given below.
`avoidance_factor` | The edge weights in A* are calculated from: `avoidance_factor*n^avoidance_k`, where `n` is the amount of customers in the end node of the edge. `n` is `0` if the node isn't visible to the customer. (Default: `1`)
`avoidance_k` | See `avoidance_factor` (Default: `1`)

Example code:
 ```python
from covid19_supermarket_abm.path_generators import get_path_generator
from covid19_supermarket_abm.utils.create_synthetic_baskets import get_all_shortest_path_dicts
from covid19_supermarket_abm.utils.node_visibility import node_visibility
import networkx as nx
entrance_nodes = [0]
till_nodes = [2]
exit_nodes = [3]
item_nodes = [1]
mu = 0.07
sigma = 0.76
weights = create_weights(G=G, data_dir=None, weight_range=10)
config['shortest_path_dict'] = get_all_shortest_path_dicts(G)
config['node_visibility'] = node_visibility(G) # An optional argument 'data_dir' can give directory of a shelves.json -file.
							# If given, visibility is calculated using these rectangular shelves.
runtime_path_generator_args = [mu, sigma, entrance_nodes, till_nodes, exit_nodes, item_nodes, weights]
path_generator_function, path_generator_args = get_path_generator(path_generation='runtime',
                                                            runtime_path_generator_args=runtime_path_generator_args)
 ```
 
 NOTE: It should be possible for the customers to get stuck in an infintite loop of avoiding each other. This hasn't happened in the kmarket-graph, but it might be more probable in other graphs.

### Node weights

For `synthetic` and `runtime` path generation, an optional weight parameter can be given in `synthetic_path_generator_args` or `runtime_path_generator_args`.
An example of how to calculate weights is given in the example code in the Runtime path generator -section. If weights are given, the item nodes are not selected uniformly, but by utilising the weights.
There are two ways to calculate weights:\
\
(1) Random weights, as used in the example code. Currently, the random weights are calculated using a uniform distribution between `1` and the given weight range (Default: `1`, i.e. uniform weights).
This is not a very interesting distribution, because the ratio of the mean and the extremes is always similar.

(2) Weights using files from directory defined in `data_dir`. If a directory is given, it needs to contain 3 files:\
-`shelves.json` containing a `product` field, giving the name for the products in that shelf. [Example](https://github.com/Saareem/gis-e4030-abm/blob/main/covid19_supermarket_abm/kmarket_data/shelves.json) \
-`products.csv` containing product names and corresponding weights. [Example](https://github.com/Saareem/gis-e4030-abm/blob/main/covid19_supermarket_abm/kmarket_data/products.csv) \
-`shelves_to_nodes.csv` linking all shelves to some nodes. [Example](https://github.com/Saareem/gis-e4030-abm/blob/main/covid19_supermarket_abm/kmarket_data/shelves_to_nodes.csv)

# Concept of general agent and inclusion of staff members
Contrary to the original program by Ying et. al, the `core.py` was reworked to include store staff members in the simulation. To enable this in least confusing way we could come up with, most of the methods that used to handle only customers were refactored to handle general (moving) agents. The methods handling agents should allow adding different kind of agents although some coding work in `core.py` is definitely required. Additionally, the method names and some attribute names have been changed correspondingly to decrease confusion. 

**Note: Inclusion of staff member will have some implications on how the results of the simulation should be interpreted, since the staff will be included in the number of susceptible, infected and total agents.** In those cases, what used to be number of customers is now the sum of customers and staff members. Correspondingly, those keys in the `results_dict` have been renamed to reflect the change.

## Staff member 
The staff member is an agent that starts its shift when the store opens and quits when it closes. This is not really realistic, but we decided it's not worth the effort to implement any kind of specific arrival and exit times, i.e. shifts as the number of staff is generally very small compared to the number of daily customers. 

The staff members start at a specified node in the store graph and move by selecting randomly a neighboring node from an **undirected copy** of the original store graph. An undirected graph is used as it was decided that the staff should be able to traverse edges to both directions. Partly this is because staff is allowed to move in the store more freely but partly also because this prevents the staff member getting trapped in a node of no return. The staff **will ignore the runtime path generation and move the same way irrespective of the path generation method**, i.e. they will not try avoiding other agents or calculating better path.

## The logic of separating different kinds of agents
As the original version didn't use object-oriented approach, we decided not to change that. However, to allow the program to handle interactions of different kinds of agents including the staff members, the `agent_id` number of the agent is used to distinguish between them. In current implementation, the staff members will populate the first `n` elements of the agent list where `n` is the amount of the staff members. Agents `id` number of which is equal or above `n` will be treated as customers in the simulation.   

Essentially if `n = 2` the agent list will be:

`agent_id` | `agent_type`
-----|------------
0    | "staff member"
1    | "staff member"
2    | "customer"
...  | ...

# Web application

Web application is in progress, but it is based on [app.py](https://github.com/Saareem/gis-e4030-abm/blob/main/app.py) - file and [flask](https://flask.palletsprojects.com/en/2.0.x/)-library for Python. The application is used mainly as a graphical user interface of original code.  

Web application contains several html-files and one css-file. Most html-files are extended from base.html. Our web application can be deployed to Azure, so requirements.txt is included. 

# Visualizations
There are three functions included for visualisation of results. As such they are meant to be used with the webapp, but they can be easily modified to be used locally as well. The functionality is based on [matplotlib.pyplot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html) API. The fuctions visualize the exposure times and contacts per node, as well as plot the variables over multiple-day simulations
 # Questions?

Original repository:
 This is work in progress, but feel free to ask any questions by raising an issue or contacting me directly under 
 [fabian.m.ying@gmail.com](fabian.m.ying@gmail.com).
 
Forked repository:
 Feel free to ask something
 [Alpo.96@gmail.com](Alpo.96@gmail.com)
 [eemeli.saarelainen@gmail.com](eemeli.saarelainen@gmail.com)
 
