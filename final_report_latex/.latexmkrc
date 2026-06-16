# Add nomenclature support to latexmk
add_cus_dep('nlo', 'nls', 0, 'makenomencl');
sub makenomencl {
  my ($base) = @_;
  return system("makeindex -s nomencl.ist -o $base.nls $base.nlo");
}