#include <stdio.h>
#include <stdlib.h>

#define DEPTH 3

struct TreeNode{
    float condition;
    int index;
    int result;
    struct TreeNode * below;
    struct TreeNode * over;
};

typedef struct TreeNode TreeNode;
typedef TreeNode * TreeNodeIndex;

TreeNodeIndex buildTree(int index[DEPTH][1 << (DEPTH - 1)], float condition[DEPTH][1 << (DEPTH - 1)], int * results, int level, int rowNumber){
    TreeNodeIndex node = (TreeNodeIndex)malloc(sizeof(TreeNode));
    if (!node) {
        return NULL;
    }
    if (level >= DEPTH){
        node->result = results[rowNumber];
        node->below = NULL;
        node->over = NULL;
        return node;
    }
    node->index = index[level][rowNumber];
    node->condition = condition[level][rowNumber];
    node->below = buildTree(index, condition, results, level+1, rowNumber*2);
    node->over = buildTree(index, condition, results, level+1, rowNumber*2+1);
    return node;
}


int getResult(float * data, TreeNodeIndex initTree){
    TreeNodeIndex node = initTree;
    while(node->below!=NULL){
        if(data[node->index]>=node->condition){
            node = node->over;
        }
        else{
            node = node->below;
        }
    }
    return node->result;
}


int main(void){
    int index[DEPTH][1 << (DEPTH-1)] = {
        {0},
        {1,2},
        {0,1,2,3}
    };
    float condition[DEPTH][1 << (DEPTH - 1)] = {
        {5},
        {6,7},
        {8,9,10,11}
    };
    int results[1 << DEPTH] = {0,1,0,1,0,1,0,1};
    TreeNodeIndex treeInit = buildTree(index, condition, results, 0, 0);
    printf("Tree Built\n");
    float data[] = {9,5,10,11};
    printf("TreeNode result: %d\n", getResult(data, treeInit));
    return 0;
}