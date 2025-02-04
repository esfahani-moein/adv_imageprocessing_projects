#include <iostream>
#include <string>
using namespace std;

#define SIZE 1000
#define M 1
#define MM -1
#define G -2

int score[SIZE][SIZE];
char path[SIZE][SIZE];
string s1, s2;

void setup_matrix(int rows, int cols) {
    for(int i = 0; i <= rows; i++) {
        score[i][0] = i * G;
        path[i][0] = 'u';
    }
    for(int j = 0; j <= cols; j++) {
        score[0][j] = j * G;
        path[0][j] = 'l';
    }
}

void calc_matrix(int rows, int cols) {
    int d, u, l;
    for(int i = 1; i <= rows; i++) {
        for(int j = 1; j <= cols; j++) {
            d = score[i-1][j-1] + (s1[i-1] == s2[j-1] ? M : MM);
            u = score[i-1][j] + G;
            l = score[i][j-1] + G;
            
            if(d >= u && d >= l) {
                score[i][j] = d;
                path[i][j] = 'd';
            }
            else if(u >= l) {
                score[i][j] = u;
                path[i][j] = 'u';
            }
            else {
                score[i][j] = l;
                path[i][j] = 'l';
            }
        }
    }
}

void find_alignment(int rows, int cols, string& out1, string& out2) {
    int i = rows, j = cols;
    
    while(i > 0 || j > 0) {
        if(i > 0 && j > 0 && path[i][j] == 'd') {
            out1 = s1[i-1] + out1;
            out2 = s2[j-1] + out2;
            i--; j--;
        }
        else if(i > 0 && path[i][j] == 'u') {
            out1 = s1[i-1] + out1;
            out2 = '-' + out2;
            i--;
        }
        else {
            out1 = '-' + out1;
            out2 = s2[j-1] + out2;
            j--;
        }
    }
}

void show_matrix(int rows, int cols) {
    cout << "\nScoring Matrix:\n  ";
    for(int j = 0; j < cols; j++)
        cout << "   " << s2[j];
    cout << endl;

    for(int i = 0; i <= rows; i++) {
        if(i == 0)
            cout << " ";
        else
            cout << s1[i-1];
        
        for(int j = 0; j <= cols; j++) {
            cout << "   " << score[i][j];
        }
        cout << endl;
    }
}

int main() {
    s1 = "GCATGCU";  // Gene Sequencing 1
    s2 = "GATTACA";  // Gene Sequencing 2
    
    int rows = s1.length();
    int cols = s2.length();
    
    if(rows > SIZE || cols > SIZE) {
        cout << "Sequence too long!" << endl;
        return 1;
    }
    
    setup_matrix(rows, cols);
    calc_matrix(rows, cols);
    
    string out1, out2;
    find_alignment(rows, cols, out1, out2);
    
    cout << "Sequence 1: " << out1 << endl;
    cout << "Sequence 2: " << out2 << endl;
    cout << "Score: " << score[rows][cols] << endl;
    show_matrix(rows, cols);
    
    return 0;
}