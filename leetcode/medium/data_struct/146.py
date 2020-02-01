# coding=utf-8

'''

哈希表+双向链表

'''

# https://leetcode-cn.com/problems/lru-cache/

class DLinkedNode(): 
    def __init__(self):
        self.key = 0
        self.value = 0
        self.prev = None
        self.next = None
            
class LRUCache():
    def _add_node(self, node):
        """
        Always add the new node right after head.
        """
        # 处理node
        node.prev = self.head
        node.next = self.head.next
        # 处理head
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node):
        """
        Remove an existing node from the linked list.
        """
        pre_node = node.prev
        next_node = node.next

        pre_node.next = next_node
        next_node.prev = pre_node

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.cache = {}
        self.size = 0
        self.capacity = capacity

        # 双向链表
        self.head, self.tail = DLinkedNode(), DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        node = self.cache.get(key, None)
        if not node:
            return -1

        # move the accessed node to the head;
        self._remove_node(node)
        self._add_node(node)

        return node.value

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        node = self.cache.get(key)

        if not node: 
            newNode = DLinkedNode()
            newNode.key = key
            newNode.value = value

            self.cache[key] = newNode
            self._add_node(newNode)

            self.size += 1

            if self.size > self.capacity:
                # pop the tail
                del_node = self.tail.prev
                self._remove_node(del_node)
                del self.cache[del_node.key]
                self.size -= 1
        else:
            # update the value.
            node.value = value
            # 插入头部
            self._remove_node(node)
            self._add_node(node)
