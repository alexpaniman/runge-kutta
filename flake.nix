{
  description = "ODE solver";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
  {

    packages.${system}.default = pkgs.clangStdenv.mkDerivation {
      name = "ode-solver";
      src = ./.;

      nativeBuildInputs = with pkgs; [
        clang-tools

        cmake
        ninja

        pyright
        (python3.withPackages (python-pkgs: [
          python-pkgs.matplotlib
        ]))
      ];

      buildInputs = with pkgs; [
        glm
        fmt
      ];

      cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Release" ];
    };

  };
}
