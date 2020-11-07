%children = ();
%visited = ();
%descendants = (); 

&readrels(\%children,0);

$counter=0;
$root="138875005";


transClos($root,\%children,\%descendants,\%visited);

printRels(\%descendants,1);


sub readrels {
   local($childhashref,$argn) = @_;
   my ($firstline,@values);
   open(ISAS,$ARGV[$argn]) || die "can't open $ARGV[$argn]";
   # read first input row
   chop($firstline = <ISAS>);
   # throw away first row, it contains the column names

   # read remaining input rows
   while (<ISAS>) {
      chop;
      @values=split('\t',$_);
      if (($values[2] eq "116680003")) { # rel.Type is "is-a" 
         $$childhashref{$values[1]}{$values[0]} = 1; # a hash of hashes, where parent is 1st arg and child is 2nd.
      }
   }
   close(ISAS);
}


#-------------------------------------------------------------------------------
# transClos
#-------------------------------------------------------------------------------
# This subroutine is based on a method described in "Transitive Closure Algorithms
# Based on Graph Traversal" by Yannis Ioannidis, Raghu Ramakrishnan, and Linda Winger,
# ACM Transactions on Database Systems, Vol. 18, No. 3, September 1993,
# Pages: 512 - 576.
# It uses a simplified version of their "DAG_DFTC" algorithm.
#-------------------------------------------------------------------------------
# 
sub transClos { # recursively depth-first traverse the graph.
   local($startnode,$children,$descendants,$visited) = @_;
   my($descendant, $childnode);
   $counter++;
   # if (($counter % 1000) eq 0) { print "Visit ", $startnode, " ", $counter, "\n"; }
   for $childnode (keys %{ $$children{$startnode} }) { # for all the children of the startnode
       unless ($$visited{$childnode}) {  # unless it has already been traversed
          &transClos($childnode,$children,$descendants,$visited); # recursively visit the childnode
          $$visited{$childnode}="T"; # and when the recursive visit completes, mark as visited
       } # end unless
       for $descendant (keys %{ $$descendants{$childnode} }) { # for each descendant of childnode
          $$descendants{$startnode}{$descendant} = 1; # mark as a descendant of startnode
       }
       $$descendants{$startnode}{$childnode} = 1; # mark the immediate childnode as a descendant of startnode
   } # end for
} # end sub transClos


#-------------------------------------------------------------------------------
# OUTPUT
#-------------------------------------------------------------------------------

sub printRels {
   local($descendants,$argn)=@_;
   open(OUTF,">$ARGV[$argn]") || die "can't open $ARGV[$argn]";
   for $startnode (keys %$descendants) {
      for $endnode ( keys %{ $$descendants{$startnode} }) {
         print OUTF "$endnode\t$startnode\n";
      }
#      print OUTF "\n";
   }
}


#-------------------------------------------------------------------------------
# END
#-------------------------------------------------------------------------------


