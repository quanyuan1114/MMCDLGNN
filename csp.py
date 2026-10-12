def countDuplicate():
    n = int(input())
    countlists = []
    for i in range(n):
        list = []
        for j in range(8):
            list.append(input().split())
        countlists.append(list)
        print(countlists.count(list))
        list.clear()


def cashCount():
    n, r = map(int, input().split())
    for i in range(n):



if __name__ == '__main__':
# countDuplicate()
