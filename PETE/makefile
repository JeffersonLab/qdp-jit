#
# Used for making the QDPOperators.h file in the include directory.
# This file is not built into the automake system.
#
all: 
	Tools/MakeOperators --classes QDPClasses.in --operators QDPOps.in \
	--pete-ops --guard QDPOPS_H --op-tags --o QDPOperators.h
	Tools/MakeOperators --operators Tools/PeteOps.in --guard PETE_PETE_OPERATORTAGS_H --op-tags --o OperatorTags.h
