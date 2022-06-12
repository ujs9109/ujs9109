prog: main.cpp
	g++ -Wall -g -o prog main.cpp

test: prog
	./prog 3 >out1.txt
	diff -abBw out1.txt ref1.txt && echo -e "\e[32mPassed test1 \e[39m"
	./prog 2 >out2.txt
	diff -abBw out2.txt ref2.txt && echo -e "\e[32mPassed test2 \e[39m"

clean:
	rm -f prog out1.txt out2.txt
