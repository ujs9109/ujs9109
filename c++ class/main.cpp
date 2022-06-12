#include <iostream>
#include <math.h>
#include <string.h>
using namespace std;

int main(int argc, char** argv)
{
    // TODO: change this to actual program.
   //int argc;
   //cin>>argc;
   argc = stoi(argv[1]);


  int size = pow(2,argc); 
 //argv[size][argc];

    for (int i = 0; i<size; i++){
        char temp[argc];
        int remainder = i ;
        int xnorcheck = 0;
        int xnornumber;

        for (int j =(argc-1); j>=0; j--) {

         int digit = pow(2,j);

         if (remainder<digit){
             temp[argc-1-j] = '0';
         }
         else{
             remainder = remainder - digit;
             temp[argc-1-j] = '1';
             xnorcheck +=1;
             }
            }

        (xnorcheck %2 ==1) ? xnornumber = 0 : xnornumber = 1;
    for(int i=0; i<argc; i++){
        cout<<temp[i];
    }
        cout<<" "<<xnornumber<< endl;

    }

    return 0;
}
