main:
	python garw.py

LN:
	KERAS_BACKEND=tensorflow python NLRWMain.py LN

NL:
	KERAS_BACKEND=tensorflow python NLRWMain.py NL

clean:
	rm -rf Output
	mkdir Output
