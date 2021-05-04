import datetime
import logging
import random
import uuid
from typing import List, Optional

import networkx as nx
import numpy as np
import simpy


class Store(object):
    """Store object that captures the state of the store"""

    def __init__(self, env: simpy.Environment, G: nx.Graph, max_customers_in_store: Optional[int] = None,
                 logging_enabled: bool = False,
                 logger: Optional[logging._loggerClass] = None, staff_conf: Optional[tuple] = (0, []),
                 realtime: Optional[bool] = False, realtime_parameters: Optional[dict] = {}):
        """
        :param env: Simpy environment on which the simulation runs
        :param G: Store graph
        :param logging_enabled: Toggle to True to log all simulation outputs
        :param max_customers_in_store: Maximum number of customers in the store
        :param staff_conf: Number of staff members in the store. Defaults to 0.
        :param path_update_freq: How often agents update their paths with realtime path generation
        :param avoidance_factor: With realtime path generation the weight of other customers in a path is calculated by
        avoidance_factor*num_customers^avoidance_k
        :param avoidance_k: With realtime path generation the weight of other customers in a path is calculated by
        avoidance_factor*num_customers^avoidance_k
        :param node_visibility: Dictionary where keys are nodes and values are nodes visible from key node. Used for
        realtime path generation.
        """
        self.n_staff = 0
        self.n_staff = max(0, staff_conf[0])  # So that nobody enters negative number
        self.realtime = realtime
        if self.realtime:
            self.path_update_freq = realtime_parameters['path_update_freq']   # How often the agents recalculate their paths when realtime
                                                    # path generation is used
            self.avoidance_factor = realtime_parameters['avoidance_factor']
            self.avoidance_k = realtime_parameters['avoidance_k']
            self.node_visibility = realtime_parameters['node_visibility']
            self.shortest_path_dict = realtime_parameters['shortest_path_dict']
        self.baskets = {}   # The nodes that the customers want to visit. Used for realtime path generation
        self.G = G.copy()
        self.agents_at_nodes = {node: [] for node in self.G}
        self.infected_agents_at_nodes = {node: [] for node in self.G}
        self.agents = []
        self.infected_agents = []
        self.env = env
        self.number_encounters_with_infected = {}
        self.number_encounters_per_node = {node: 0 for node in self.G}
        self.arrival_times = {}
        self.exit_times = {}
        self.shopping_times = {}
        self.waiting_times = {}
        self.agents_next_zone = {}  # maps agent to the next zone that it wants to go
        self.is_open = True
        self.is_closed_event = self.env.event()
        self.time_with_infected_per_agent = {}
        self.time_with_infected_per_node = {node: 0 for node in self.G}
        self.node_arrival_time_stamp = {}
        self.num_customers_waiting_outside = 0
        self.total_time_crowded = 0
        self.crowded_thres = 4
        self.node_is_crowded_since = {node: None for node in self.G}  # is None if not crowded, else it's the start time
        self.logs = []
        self.logging_enabled = logging_enabled
        self.logger = logger
        self.staff_paths = {id: [] for id in range(0, self.n_staff)}

        # Parameters
        self.node_capacity = np.inf
        self.with_node_capacity = False
        if max_customers_in_store is None:
            self.max_customers_in_store = np.inf
        else:
            self.max_customers_in_store = int(max_customers_in_store)
        # if logger is None:
        #     self.logging_enabled = False
        # else:
        #     self.logging_enabled = True
        self.counter = simpy.Resource(self.env, capacity=self.max_customers_in_store)

        # Stats recording
        self.stats = {}

    def open_store(self):
        assert self.get_customer_count() == 0, "Customers are already in the store before the store is open"
        self.is_open = True

    def close_store(self):
        self.log(f'Store is closing. There are {self.number_customers_in_store()} left in the store. ' +
                 f'({self.num_customers_waiting_outside} are waiting outside)')
        self.is_open = False
        self.is_closed_event.succeed()

    def enable_node_capacity(self, node_capacity: int = 2):
        self.with_node_capacity = True
        self.node_capacity = node_capacity

    def number_customers_in_store(self):
        return sum([len(cus) for cus in list(self.agents_at_nodes.values())]) - self.n_staff

    def get_customer_count(self) -> int:
        return len(self.agents[self.n_staff:])

    def move_agent(self, agent_id: int, infected: bool, start: int, end: int) -> bool:
        agent_type = "Customer"
        msg_dict = {"Customer": "stays at present location to buy something.",
                    "Staff member": "stays at present location to organize shelves."}
        if agent_id < self.n_staff:
            agent_type = "Staff member"
        if self.check_valid_move(start, end):
            if start == end:  # start == end
                self._agent_wait(agent_id, start, infected)
                if start in self.baskets[agent_id]:
                    self.baskets[agent_id].remove(start)
                self.log(f'{agent_type} {agent_id} {msg_dict[agent_type]}')
                has_moved = True
            elif self.with_node_capacity and len(self.agents_at_nodes[end]) >= self.node_capacity \
                    and start not in [self.agents_next_zone[agent] for agent in self.agents_at_nodes[end]]:
                # Wait if next node is occupied and doesn't work.
                self.log(f'{agent_type} {agent_id} is waiting at {start}, ' +
                         f'since the next node {end} is full. [{self.agents_at_nodes[end]}]')
                self._agent_wait(agent_id, start, infected)
                has_moved = False
            else:
                self.log(f'{agent_type} {agent_id} is moving from {start} to {end}.')
                self._agent_departure(agent_id, start, infected)
                self._agent_arrival(agent_id, end, infected)
                has_moved = True
        else:
            raise ValueError(f'{start} -> {end} is not a valid transition in the graph!')
        return has_moved

    def check_valid_move(self, start: int, end: int, oneway: bool = False):
        """
        Checks if the move from start to end is a valid edge in the store graph.
        @param start: start node
        @param end: end node
        @param oneway: flag for whether validity is checked for both directions
        @return: the validity to traverse the edge from start to end (and back, if oneway=True)
        """
        validity = False
        if self.G.has_edge(start, end) or start == end:
            validity = True
            if oneway:
                validity = validity and self.G.has_edge(end, start)
        return validity

    def add_agent(self, agent_id: int, start_node: int, infected: bool, wait: float = 0, basket: Optional[List[int]] = []):
        """
        Adds an agent into the store. The agent can currently be either staff member or a customer. A staff member will
        have id < n_staff if n_staff != 0 and customers will have id >= n_staff
        @param agent_id: ID of the agent that is added
        @param start_node: the node where the agent starts
        @param infected: Whether the agent is infected or not
        @param wait: Optional
        """
        self.arrival_times[agent_id] = self.env.now
        if agent_id < self.n_staff:
            self.log(f'New staff member {agent_id} starts shift. ' +
                     f'({infected * "infected"}{(not infected) * "susceptible"})')
            self.waiting_times[agent_id] = 0
        else:
            self.log(f'New customer {agent_id} arrives at the store. ' +
                     f'({infected * "infected"}{(not infected) * "susceptible"})')
            self.waiting_times[agent_id] = wait
        self.agents.append(agent_id)
        self.baskets[agent_id] = basket
        if not infected:
            # Increase counter
            self.number_encounters_with_infected[agent_id] = 0
            self.time_with_infected_per_agent[agent_id] = 0
        else:
            self.infected_agents.append(agent_id)
        self._agent_arrival(agent_id, start_node, infected)

    def add_customer(self, customer_id: int, start_node: int, infected: bool, wait: float):
        """
        Adds customer to the store. Exists for background compatibility reasons.
        @param customer_id: The ID of the customer
        @param start_node: The node where the customer starts
        @param infected: Whether the customer is infected or not
        @param wait: time to wait outside the store in the que if the store capacity is reached
        """
        self.add_agent(self, customer_id, start_node, infected, wait)

    def _infect_other_agents_at_node(self, agent_id: int, node: int):
        agent_type = "staff member" if agent_id < self.n_staff else "customer"
        other_susceptible_agents = [other_agent for other_agent in self.agents_at_nodes[node] if
                                    other_agent not in self.infected_agents_at_nodes[node]]
        if len(other_susceptible_agents) > 0:
            self.log(
                f'Infected {agent_type} {agent_id} arrived in {node} and' +
                f' met {len(other_susceptible_agents)} agents')
        for other_agent in other_susceptible_agents:
            self.number_encounters_with_infected[other_agent] += 1
            self.number_encounters_per_node[node] += 1

    def _get_infected_by_other_agents_at_node(self, agent_id: int, node: int):
        agent_type = "Staff member" if agent_id < self.n_staff else "Customer"
        num_infected_here = len(self.infected_agents_at_nodes[node])

        # Track number of infected agents
        if num_infected_here > 0:
            self.log(
                f'{agent_type} {agent_id} is in at zone {node} with {num_infected_here} infected people.' +
                f' ({self.infected_agents_at_nodes[node]})')
            self.number_encounters_with_infected[agent_id] += num_infected_here
            self.number_encounters_per_node[node] += num_infected_here

    def _agent_arrival(self, agent_id: int, node: int, infected: bool):
        """Process an agent arriving at a node."""
        self.agents_at_nodes[node].append(agent_id)
        self.node_arrival_time_stamp[agent_id] = self.env.now
        if infected:
            self.infected_agents_at_nodes[node].append(agent_id)
            self._infect_other_agents_at_node(agent_id, node)
        else:
            self._get_infected_by_other_agents_at_node(agent_id, node)
        num_agents_at_node = len(self.agents_at_nodes[node])
        if num_agents_at_node >= self.crowded_thres and self.node_is_crowded_since[node] is None:
            self.log(f'Node {node} has become crowded with {num_agents_at_node} agents here.')
            self.node_is_crowded_since[node] = self.env.now

    def _agent_wait(self, customer_id: int, node: int, infected: bool):
        if infected:
            self._infect_other_agents_at_node(customer_id, node)
        else:
            self._get_infected_by_other_agents_at_node(customer_id, node)

    def _agent_departure(self, agent_id: int, node: int, infected: bool):
        """
        Process an agent departing from a node.
        @param agent_id: ID of the agent
        @param node: node which is left
        @param infected: whether the agent is infected or not
        """
        self.agents_at_nodes[node].remove(agent_id)
        if infected:
            self.infected_agents_at_nodes[node].remove(agent_id)
            s_agents = self.get_susceptible_agents_at_node(node)
            for s_agent in s_agents:
                dt_with_infected = self.env.now - max(self.node_arrival_time_stamp[s_agent],
                                                      self.node_arrival_time_stamp[agent_id])
                self.time_with_infected_per_agent[s_agent] += dt_with_infected
                self.time_with_infected_per_node[node] += dt_with_infected
        else:
            i_agents = self.infected_agents_at_nodes[node]
            for i_agent in i_agents:
                dt_with_infected = self.env.now - max(self.node_arrival_time_stamp[i_agent],
                                                      self.node_arrival_time_stamp[agent_id])
                self.time_with_infected_per_agent[agent_id] += dt_with_infected
                self.time_with_infected_per_node[node] += dt_with_infected

        num_agents_at_node = len(self.agents_at_nodes[node])
        if self.node_is_crowded_since[node] is not None and num_agents_at_node < self.crowded_thres:
            # Node is no longer crowded
            total_time_crowded_at_node = self.env.now - self.node_is_crowded_since[node]
            self.total_time_crowded += total_time_crowded_at_node
            self.log(
                f'Node {node} is no longer crowded ({num_agents_at_node} agents here. ' +
                f'Total time crowded: {total_time_crowded_at_node:.2f}')
            self.node_is_crowded_since[node] = None

    def get_susceptible_agents_at_node(self, node):
        return [a for a in self.agents_at_nodes[node] if a not in self.infected_agents_at_nodes[node]]

    def remove_agent(self, agent_id: int, last_position: int, infected: bool):
        """
        Removes agent from the store and saves the exit time, node arrival time stamp and shopping times if the agent
        is a customer.
        @param agent_id: ID of the agent. if ID < n_staff, the agent is member of staff
        @param last_position: The exit node
        @param infected: whether the agent is infected or not
        """
        agent_type = "Staff member" if agent_id < self.n_staff else "Customer"
        self._agent_departure(agent_id, last_position, infected)
        self.exit_times[agent_id] = self.env.now
        self.node_arrival_time_stamp[agent_id] = self.env.now
        self.shopping_times[agent_id] = self.exit_times[agent_id] - self.arrival_times[agent_id]
        self.log(f'{agent_type} {agent_id} left the store.')

    def now(self):
        return f'{self.env.now:.4f}'

    def log(self, string: str):
        if self.logging_enabled:
            self.logs.append(f'[Time: {self.now()}] ' + string)
        if self.logger is not None:
            self.logger.debug(f'[Time: {self.now()}] ' + string)

    def update_path(self, start: int, agent_id: int):
        """
        Updates path for an agent when realtime path generation is used.

        :param start: Current location of the agent
        :param agent_id: ID of the agent
        """

        def _weight_function(u, v, e):
            """
            Function used to calculate edge weights in the graph
            """
            agents_in_node = self.node_visibility[(start, v)]*len(self.agents_at_nodes[v])
            return 1 + self.avoidance_factor*agents_in_node**self.avoidance_k

        def _heuristic_function(source, target):
            # TODO At the moment shortest_path_dict needs to be in config, should be fixed
            return len(self.shortest_path_dict[source][target][0])

        shortest_path = [start]
        shortest_len = float("inf")
        not_visited = self.baskets[agent_id]

        if len(not_visited) > 0:
            path = nx.algorithms.astar_path(self.G, start, not_visited[0], heuristic=_heuristic_function,
                                                        weight=_weight_function)
            N = min(self.path_update_freq, len(path))
            shortest_path = path[:N]

        # Duplicate nodes in basket so the agent stays in them to buy something
        if shortest_path[-1] in not_visited:
            shortest_path.append(shortest_path[-1])

        return shortest_path


def customer(env: simpy.Environment, customer_id: int, infected: bool, store: Store, path_orig: List[int],
             traversal_time: float, thres: int = 50):
    """
    Simpy process simulating a single customer

    :param env: Simpy environment on which the simulation runs
    :param customer_id: ID of customer
    :param infected: True if infected
    :param store: Store object
    :param path_orig: Assigned customer shopping path (for non-realtime path generation) or item basket (realtime)
    :param traversal_time: Mean time before moving to the next node in path (also called waiting time)
    :param thres: Threshold length of queue outside. If queue exceeds threshold, customer does not enter
    the queue and leaves.
    """

    path = []
    if store.realtime:
        # Agent starts at entrance node
        path = [path_orig[1], path_orig[1]]
    else:
        path = path_orig

    arrive = env.now

    if store.num_customers_waiting_outside > thres:
        store.log(f'Customer {customer_id} does not queue up, since we have over {thres} customers waiting outside ' +
                  f'({store.num_customers_waiting_outside})')
        return
    else:
        store.num_customers_waiting_outside += 1

    with store.counter.request() as my_turn_to_enter:
        result = yield my_turn_to_enter | store.is_closed_event
        store.num_customers_waiting_outside -= 1
        wait = env.now - arrive

        if my_turn_to_enter not in result:
            store.log(f'Customer {customer_id} leaves the queue after waiting {wait:.2f} min, as shop is closed')
            return

        if my_turn_to_enter in result:
            store.log(
                f'Customer {customer_id} enters the shop after waiting {wait :.2f} min with shopping path {path}.')
            start_node = path[0]
            basket = []
            if store.realtime:
                basket = path_orig[1:]
            store.add_agent(customer_id, start_node, infected, wait, basket=basket)
            while len(path) > 1:
                for start, end in zip(path[:-1], path[1:]):
                    store.agents_next_zone[customer_id] = end
                    has_moved = False
                    while not has_moved:  # If it hasn't moved, wait a bit
                        yield env.timeout(random.expovariate(1 / traversal_time))
                        has_moved = store.move_agent(customer_id, infected, start, end)
                path = store.update_path(path[-1], customer_id)
            yield env.timeout(random.expovariate(1 / traversal_time))  # wait before leaving the store
            store.remove_agent(customer_id, path[-1], infected)


def staff_member(env: simpy.Environment, staff_id: int, infected: bool, store: Store, traversal_time: float,
                 path: List[int]):
    """
    Simpy process simulating a member of staff

    :param env: Simpy environment on which the simulation runs
    :param staff_id: ID of agent
    :param infected: True if infected
    :param store: Store object
    :param traversal_time: Mean time before moving to the next node in path (also called waiting time)
    """
    for start, end in zip(path[:-1], path[1:]):
        store.agents_next_zone[staff_id] = end
        has_moved = False
        while not has_moved:  # If it hasn't moved, wait a bit
            yield env.timeout(random.expovariate(1 / traversal_time))
            has_moved = store.move_agent(staff_id, infected, start, end)
    yield env.timeout(random.expovariate(1 / traversal_time))  # wait before leaving the store
    store.remove_agent(staff_id, path[-1], infected)


def two_customers(env: simpy.Environment, customer_id: int, infected: bool, store: Store, path_orig: List[int],
                  traversal_time: float, thres: int = 50):
    """
    Simpy process simulating group of TWO customers

    :param env: Simpy environment on which the simulation runs
    :param customer_id: ID of customer
    :param infected: True if infected
    :param store: Store object
    :param path: Assigned customer shopping path
    :param traversal_time: Mean time before moving to the next node in path (also called waiting time)
    :param thres: Threshold length of queue outside. If queue exceeds threshold, customer does not enter
    the queue and leaves.
    """

    realtime = False
    path = []
    if store.realtime:
        # Agent starts at entrance node
        path = [path_orig[1], path_orig[1]]
    else:
        path = path_orig

    arrive = env.now

    if store.num_customers_waiting_outside > thres:
        store.log(f'Customers {customer_id} and {customer_id + 1} does not queue up, since we have over' +
                  f' {thres} customers waiting outside ' + f'({store.num_customers_waiting_outside})')
        return
    else:
        store.num_customers_waiting_outside += 2

    with store.counter.request() as my_turn_to_enter:
        result = yield my_turn_to_enter | store.is_closed_event
        store.num_customers_waiting_outside -= 2
        wait = env.now - arrive

        if my_turn_to_enter not in result:
            store.log(f'Customers {customer_id} and {customer_id + 1} leave the queue after waiting +'
                      f'{wait:.2f} min, as shop is closed')
            return

        if my_turn_to_enter in result:
            store.log(f'Customers {customer_id} and {customer_id + 1} enter the shop after waiting +'
                      f'{wait :.2f} min with shopping path {path}.')
            start_node = path[0]
            basket = []
            if realtime:
                basket = path_orig[1:]
            store.add_agent(customer_id, start_node, infected, wait, basket=basket)
            store.add_agent(customer_id + 1, start_node, infected, wait, basket=basket)
            while len(path) > 1:
                for start, end in zip(path[:-1], path[1:]):
                    store.agents_next_zone[customer_id] = end
                    store.agents_next_zone[customer_id + 1] = end
                    has_moved = False
                    has_moved1 = False
                    while not has_moved and not has_moved1:  # If they haven't moved, wait a bit
                        yield env.timeout(random.expovariate(1 / traversal_time))
                        has_moved = store.move_agent(customer_id, infected, start, end)
                        has_moved1 = store.move_agent(customer_id + 1, infected, start, end)
                path = store.update_path(path[-1], customer_id)
            yield env.timeout(random.expovariate(1 / traversal_time))  # wait before leaving the store
            store.remove_agent(customer_id, path[-1], infected)
            store.remove_agent(customer_id + 1, path[-1], infected)


def _stats_recorder(store: Store):
    store.stats['num_customers_in_store'] = {}
    env = store.env
    while store.is_open or store.number_customers_in_store() > 0:
        store.stats['num_customers_in_store'][env.now] = store.number_customers_in_store()
        yield env.timeout(10)


def _agent_arrivals(env: simpy.Environment, store: Store, path_generator, config: dict, popular_hours,
                    num_hours_open, n_staff: Optional[int] = 0):
    """Process that creates all agents."""
    hour_nro = 0
    arrival_rate = config['arrival_rate'] * popular_hours[hour_nro] / popular_hours.mean()
    infection_proportion = config['infection_proportion']
    traversal_time = config['traversal_time']
    if 'customers_together' not in config:
        config['customers_together'] = 1e-12  # TODO: Eemeli: Consider fixing this in other way.
    customer_id = 0
    store.open_store()
    yield env.timeout(random.expovariate(arrival_rate))
    while env.now < num_hours_open * 60:
        infected = np.random.rand() < infection_proportion
        path = path_generator.__next__()

        # Some customers arrive together
        if config['customers_together'] != 0 and customer_id % int(1 / config['customers_together']) == 0:
            env.process(two_customers(env, customer_id, infected, store, path, traversal_time))
            customer_id += 2
        else:
            env.process(customer(env, customer_id, infected, store, path, traversal_time))
            customer_id += 1

        # Change arrival rate every hour
        if env.now / 60 >= hour_nro + 1:
            hour_nro += 1
            arrival_rate = config['arrival_rate'] * popular_hours[hour_nro] / popular_hours.mean()
        yield env.timeout(random.expovariate(arrival_rate))
    store.close_store()


def _sanity_checks(store: Store,
                   # logger: Optional[logging._loggerClass] = None, log_capture_string=None,
                   raise_test_error=False):
    infectious_contacts_list = [i for i in store.number_encounters_with_infected.values() if i != 0]
    num_susceptible = len(store.number_encounters_with_infected)
    num_infected = len(store.infected_agents)
    num_cust = len(store.agents)

    try:
        assert sum(infectious_contacts_list) == sum(store.number_encounters_per_node.values()), \
            "Number of infectious contacts doesn't add up"
        assert num_infected + num_susceptible == num_cust, \
            "Number of infected and susceptible customers doesn't add up to total number of customers"

        customers_at_nodes = [len(val) for val in store.infected_agents_at_nodes.values()]
        assert max(customers_at_nodes) == 0, \
            f"{sum(customers_at_nodes)} customers have not left the store. {store.infected_agents_at_nodes}"
        # TODO: Eemeli: Will most likely fail with general agent. Fix test if it's used.
        assert max([len(val) for val in store.agents_at_nodes.values()]) == 0, \
            f"{sum(customers_at_nodes)} customers have not left the store. {store.agents_at_nodes}"
        assert set(store.waiting_times.keys()) == set(store.agents), \
            'Some customers are not recorded in waiting times (or vice versa)'
        assert all([val >= 0 for val in store.waiting_times.values()]), \
            'Some waiting times are negative!'
        actual_max_customer_in_store = max(store.stats['num_customers_in_store'].values())
        assert actual_max_customer_in_store <= store.max_customers_in_store, \
            f'Somehow more people were in the store than allowed ' + \
            f'(Allowed: {store.max_customers_in_store} | Actual: {actual_max_customer_in_store})'

        assert store.num_customers_waiting_outside == 0, \
            f"Somehow, there are still {store.num_customers_waiting_outside} people waiting outside"
        if raise_test_error:
            raise RuntimeError("Test error")
    except Exception as e:
        time_string = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_name = f'log_{time_string}_{uuid.uuid4().hex}.log'
        print(f'Sanity checks NOT passed. Something went wrong. Saving logs in {log_name}.')
        with open(log_name, 'w') as f:
            f.write('\n'.join(store.logs))
        if not raise_test_error:
            raise e
    if store.logger is not None:
        store.logger.info('Sanity checks passed!')
