from typing import Optional


class ListNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        # 创建虚拟头部节点以简化删除过程
        dummy_head = ListNode(val = 55,next = head)
        # 遍历列表并删除值为val的节点
        current = dummy_head
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
            else:
                current = current.next
        
        return dummy_head.next
    
def creat_list_node(elements):
    dummy = ListNode(0)
    current = dummy
    for element in elements:
        current.next = ListNode(element)
        current = current.next
    return dummy.next

def print_list_node(head):
    current = head
    elements = []
    while current:
        elements.append(current.val)
        current = current.next
    print(elements)

a = creat_list_node([1,2,3,4,5])
print_list_node(a)

solution = Solution()
b = solution.removeElements(a, 3)
print_list_node(b)