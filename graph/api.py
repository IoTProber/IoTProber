import re
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import logging
import datetime
import pandas as pd
import numpy as np
from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher, database, cypher

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ProtocolGraph:
    def __init__(self, url: str, username: str, password: str):
        self.graph_db = Graph(url, auth=(username, password))
        self.os = os.name  # Can be Unix/Linux(posix), Windows(nt), macOS(posix)

    def Clear(self):
        self.graph_db.delete_all()

    def CypherQuery(self, query: str, *params):
        placeholders = re.findall(r'\$p(\d+)', query)

        if not placeholders:
            expected_count = 0
        else:
            # Find the maximum number of placeholders we expect to have the same number of parameters.
            indices = {int(p) for p in placeholders}
            expected_count = max(indices)

            # (Optional) Stricter checking: ensure placeholders are contiguous
            if set(range(1, expected_count + 1)) != indices:
                raise ValueError(
                    f"Query placeholders are not sequential. Found indices {sorted(list(indices))}, "
                    f"but expected a continuous sequence from 1 to {expected_count}."
                )

        if len(params) != expected_count:
            raise ValueError(
                f"Query expects {expected_count} parameter(s) (based on placeholder ${expected_count}), "
                f"but {len(params)} were provided."
            )

        param_dict = {f'p{i + 1}': params[i] for i, _ in enumerate(params)}

        log.info(f"[Cypher Query]: {query} with parameters: {param_dict}")

        try:
            results = self.graph_db.run(query, **param_dict).data()
            return results
        except Exception as e:
            log.error(f"An error occurred during query execution: {e}")
            return None

    def Merge(self, label: str, properties: dict):
        cypher_query = f"MERGE (n:{label} {{ "
        prop_list = [f"{key}: ${key}" for key in properties.keys()]
        cypher_query += ", ".join(prop_list)
        cypher_query += " }) RETURN n"

        self.graph_db.run(cypher_query)

    def MatchSingleNode(self, labels, attr):
        matcher = NodeMatcher(self.graph_db)
        matching = ""
        count = 0
        for key in attr:
            if count != 0:
                matching += " and "
            matching += "_." + str(key) + "=\'" + str(attr[key]) + "\'"
            count += 1
        # py2neo's NodeMatcher.match signature is match(*labels, **properties).
        # If we pass a list directly, it may be treated as a single label object
        # and cause issues like TypeError: unhashable type: 'list'.
        if isinstance(labels, (list, tuple, set)):
            result = matcher.match(*labels).where(matching).first()
        else:
            result = matcher.match(labels).where(matching).first()
        return result

    def MatchMultipleNode(self, labels, attr):
        matcher = NodeMatcher(self.graph_db)
        matching = ""
        for index, key in enumerate(attr):
            if index != 0:
                matching += " and "
            matching += "_." + str(key) + "=\'" + str(attr[key]) + "\'"
        if isinstance(labels, (list, tuple, set)):
            result = matcher.match(*labels).where(matching).all()
        else:
            result = matcher.match(labels).where(matching).all()
        return result

    def CreateRelationshipAttr(self, label1, attr1, label2, attr2, relation_label, relation_list):
        node1 = self.MatchSingleNode(label1, attr1)
        node2 = self.MatchSingleNode(label2, attr2)
        if node1 is None or node2 is None:
            return False
        relationship = Relationship.type(relation_label)
        events = relationship(node1, node2)

        for key, value in relation_list.items():
            events[key] = value

        log.info("[+] Creating Relationship: {}".format(events))
        result = self.graph_db.create(events)
        return result

    def CreateRelationship(self, node1: Node, node2: Node, relation_label: str, relation_attr: dict):
        if node1 is None or node2 is None:
            return False
        relationship = Relationship.type(relation_label)
        events = relationship(node1, node2)

        for key, value in relation_attr.items():
            events[key] = value

        total_count = self.Count()
        # log.info("[+] Creating Relationship: {}".format(total_count))

        result = self.graph_db.create(events)
        return result

    def MatchRelationship(self, relation_label: str, relation_attr: dict):
        judge_condition = ""
        index = 0
        for key, value in relation_attr.items():
            if type(value) == str:
                judge_condition += "r.{}='{}'".format(key, value)
            else:
                judge_condition += "r.{}={}".format(key, value)
            if index != len(relation_attr) - 1:
                judge_condition += " and "
            index += 1

        relation_label = relation_label.replace(" ", "_")
        cypher_query = "MATCH (a)-[r:{}]->(b) WHERE {} RETURN a,r,b".format(relation_label, judge_condition)
        # print(cypher_query)
        relationship = self.graph_db.run(cypher_query).data()
        return relationship

    def MatchRelationship2(self, node_attr: dict, relation_label: str, relation_attr: dict):
        judge_condition = ""
        for key, value in node_attr.items():
            if type(value) == str:
                judge_condition += "a.{}='{}'".format(key, value)
            else:
                judge_condition += "a.{}={}".format(key, value)
            judge_condition += " and "

        index = 0
        for key, value in relation_attr.items():
            if type(value) == str:
                judge_condition += "r.{}='{}'".format(key, value)
            else:
                judge_condition += "r.{}={}".format(key, value)
            if index != len(relation_attr) - 1:
                judge_condition += " and "
            index += 1

        relation_label = relation_label.replace(" ", "_")
        cypher_query = "MATCH (a)-[r:{}]->(b) WHERE {} RETURN a,r,b".format(relation_label, judge_condition)
        # print(cypher_query)
        relationship = self.graph_db.run(cypher_query).data()
        return relationship

    def CreateNode(self, labels: list, properties: dict):
        """
        Finds a node with the given labels and properties, or creates it if it doesn't exist.
        This function uses the MERGE Cypher statement for efficiency and atomicity.

        Args:
            graph: The py2neo.Graph instance.
            labels: A list of strings for the node's labels. E.g., ['Person', 'Customer'].
            properties: A dictionary of the node's properties used for matching and creating.
                        E.g., {'name': 'Alice', 'email': 'alice@example.com'}.

        Returns:
            A tuple containing:
            (node: py2neo.Node, was_created: bool)
            - node: The found or newly created node.
            - was_created: True if the node was newly created, False if it already existed.
        """
        if not labels:
            raise ValueError("Labels list cannot be empty.")
        if not properties:
            raise ValueError("Properties dictionary cannot be empty.")

        # 1. Dynamically construct the MERGE query
        # Format labels: ['Person', 'Customer'] -> :Person:Customer
        label_str = ":" + ":".join(labels)

        # Format properties for the MERGE clause: {name: $name, email: $email}
        # This uses parameters, which is the best practice.
        prop_str = ", ".join([f"{key}: ${key}" for key in properties.keys()])

        # We use a trick with ON CREATE to determine if the node was created or matched.
        # The query returns the node and a boolean flag `wasCreated`.
        query = f"""
        MERGE (n{label_str} {{ {prop_str} }})
        ON CREATE SET n._was_created_marker = true
        WITH n, n._was_created_marker IS NOT NULL AS wasCreated
        REMOVE n._was_created_marker
        RETURN n, wasCreated
        """

        try:
            # 2. Execute the query using the properties dictionary as parameters
            result = self.graph_db.run(query, **properties).data()

            if result:
                data = result[0]
                node = data['n']
                was_created = data['wasCreated']
                return node, was_created
            else:
                # This case should theoretically not happen with MERGE
                return None, False

        except Exception as e:
            log.error(f"An error occurred: {e}")
            return None, False

    def MergeNode(self, graph: Graph, nodes_to_merge: list, config: dict = None):
        """
        Merges a list of nodes into a single node using the APOC procedure.

        The first node in the list is treated as the 'survivor'.

        Args:
            graph: The py2neo.Graph instance to connect to the database.
            nodes_to_merge: A list of py2neo.Node objects to be merged.
            config: A dictionary for the APOC merge configuration.json.
                    Defaults to combining properties.
                    Example: {'properties': 'combine', 'mergeRels': True}

        Returns:
            The surviving py2neo.Node object after the merge, or None if failed.
        """
        if not nodes_to_merge or len(nodes_to_merge) < 2:
            print("Error: At least two nodes are required for a merge.")
            return None

        # Default APOC configuration.json
        if config is None:
            config = {'properties': 'combine', 'mergeRels': True}

        # Get the unique IDs of the nodes to merge
        node_ids = [node.identity for node in nodes_to_merge]

        # Construct the Cypher query using parameterized query
        query = """
        MATCH (n) WHERE id(n) IN $ids
        // Ensure the nodes are processed in the original order
        WITH n, apoc.coll.indexOf($ids, id(n)) as idx
        ORDER BY idx
        WITH collect(n) AS nodes
        CALL apoc.refactor.mergeNodes(nodes, $config) YIELD node
        RETURN node
        """

        try:
            result = graph.run(query, ids=node_ids, config=config).data()
            if result:
                # The query returns the merged node
                surviving_node = result[0]['node']
                log.info(f"Successfully merged {len(nodes_to_merge)} nodes into node with ID: {surviving_node.identity}")
                return surviving_node
            else:
                log.info("Merge operation did not return a node.")
                return None
        except Exception as e:
            log.error(f"An error occurred during the APOC merge: {e}")
            return None

    def Count(self):
        cypher_query = "MATCH (a)-[r]->(b) RETURN count(r)"
        count = self.graph_db.run(cypher_query).data()[0]["count(r)"]
        return count

    def SingleCount(self, device: str):
        cypher_query = 'MATCH (a)-[r]->(b) WHERE r.device="{}" RETURN count(r)'.format(device)
        count = self.graph_db.run(cypher_query).data()[0]["count(r)"]
        return count

    def GraphShow(self, device, phase, total, single, total_message, single_message):
        if self.os == "nt":
            os.system("cls")
        else:
            os.system('clear')

        print("\r \t\t\tProcessing {}({})                        \n"
              "\r┏━━━━━━━━━━━━━━━━━━━━━━┓━━━━━━━━━━━━━━━━━━━━━━┓━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓━━━━━━━━━━━━━━━━━━━━━━┓\n"
              "\r┃    Total: {:<5}      ┃     Count: {:<5}     ┃    Total Messages: {:<5}     ┃    Messages: {:<5}   ┃\n"
              "\r┗━━━━━━━━━━━━━━━━━━━━━━┛━━━━━━━━━━━━━━━━━━━━━━┛━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛━━━━━━━━━━━━━━━━━━━━━━┛\n".format(
                device, phase, total, single, total_message, single_message), end='', flush=True)


if __name__ == "__main__":
    # graph_db_url = "neo4j://localhost:7687"
    graph_db_url = "http://localhost:7474"
    graph = ProtocolGraph(graph_db_url, "neo4j", "avs01046")
    count = graph.Count()
