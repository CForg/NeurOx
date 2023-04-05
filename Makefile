include Makefile.inc

INC := .\include
UPINC := ..\include
DOC := ./docs
OxMenu = ..\OxMenu\include

vpath %.ox $(INC)
vpath %.h $(INC)
vpath %.oxh $(INC)
vpath %.oxo $(INC)
vpath %.ox.html $(DOC)

NeurOxobjects = Network.oxo Layers.oxo 

NeurOx.oxo  : $(NeurOxobjects)

%.oxo : %.ox %.oxh
	$(OX) $(OXFLAGS) -i$(UPINC);$(INC) -d -4 $<
#	$(COPY) source\$@ $(INC)
#	$(ERASE) source\$@

.PHONY : document
document:
	$(ERASE) ${DOC}\default.html
	${MAKE} -C $(DOC) tweak
	$(COPY) ..\docs\index.tmp ..\docs\default.html
	$(COPY) ..\docs\index.tmp ..\docs\index.html

