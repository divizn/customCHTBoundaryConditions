/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
/**
 * @file CoupledBoundary.H
 * @author B McCormick
 * @date 3 March 2026
 * @brief Header file for CoupledBoundary.C
 */

#ifndef CoupledBoundary_H
#define CoupledBoundary_H

#include "mixedFvPatchFields.H"
#include "temperatureCoupledBase.H"
#include "scalarField.H"
#include "mui.h"
#include "muiconfig.h"
#include "fvPatchFieldMapper.H"
#include "volFields.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
/*---------------------------------------------------------------------------*\
   Class CoupledBoundary Declaration
\*---------------------------------------------------------------------------*/

class CoupledBoundary
:
    public mixedFvPatchScalarField,
    public temperatureCoupledBase
{
// Private data
private:
    // initialise samplers
    mui::sampler_pseudo_n2_linear<mui::mui_config> spatial_sampler{0.2};
    std::unique_ptr<mui::sampler_rbf<mui::mui_config>> rbf_sampler;
    word sampler_name = "pseudo_n2_linear";
    scalar rbf_radius = 0.3;
    label rbf_basis = 0;
    mui::temporal_sampler_linear<mui::mui_config> temporal_sampler;

    // push and fetch points
    mui::point3d push_point;
    mui::point3d fetch_point;

    // initialise interface
    std::vector<std::unique_ptr<mui::uniface<mui::mui_config>>> mui_ifs;
    
    // MPI rank
    int rank;

    // initialise iteration
    double iteration=1;  

    
public:
    // Constructors

        //- Construct from patch and internal field
        CoupledBoundary
        (
            const fvPatch& p,
            const DimensionedField<scalar, volMesh>& iF
        )
        :
            mixedFvPatchScalarField(p, iF),
            temperatureCoupledBase(patch())
        {
            this->refValue() = Zero;
            this->refGrad() = Zero;
            this->valueFraction() = 1.0;
        }

        //- Construct from patch, internal field and dictionary
        CoupledBoundary
        (
            const CoupledBoundary& ptf,
            const fvPatch& p,
            const DimensionedField<scalar, volMesh>& iF,
            const fvPatchFieldMapper& mapper
        )
        :
            mixedFvPatchScalarField(ptf, p, iF, mapper),
            temperatureCoupledBase(patch(), ptf)
        {}

        CoupledBoundary
        (
            const fvPatch& p,
            const DimensionedField<scalar, volMesh>& iF,
            const dictionary& dict
        )
        :
            mixedFvPatchScalarField(p, iF),
            temperatureCoupledBase(patch(), dict)
        {   
            // get MPI rank
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);


            // Create a new communicator of only ranks along the interface
            MPI_Comm interface_ranks;
            int colour = 0;
            int key;
            if (this->size() > 0){
               colour = 1;
            }
            MPI_Comm_split(MPI_COMM_WORLD, colour, rank, &interface_ranks);

            if(this->size() > 0){
                // intialise MUI interface
                std::vector<std::string> ifsName;
                ifsName.emplace_back("CoupledBoundary");
                mui_ifs=mui::create_uniface<mui::mui_config>( "OpenFOAM", ifsName, interface_ranks);

                sampler_name = dict.getOrDefault<word>("spatialSampler", "pseudo_n2_linear");
                rbf_radius = dict.getOrDefault<scalar>("rbfRadius", 0.3);
                rbf_basis = dict.getOrDefault<label>("rbfBasis", 0);
                if (sampler_name == "rbf")
                {
                    // RBF needs fetch points up front
                    std::vector<mui::point3d> pts;
                    const vectorField& Cf = p.Cf();
                    forAll(Cf, faceI)
                    {
                        mui::point3d pt;
                        pt[0] = round_to(Cf[faceI].x(), 8);
                        pt[1] = round_to(Cf[faceI].y(), 8);
                        pt[2] = round_to(Cf[faceI].z(), 8);
                        pts.push_back(pt);
                    }
                    rbf_sampler = std::make_unique<mui::sampler_rbf<mui::mui_config>>
                    (
                        rbf_radius, pts, rbf_basis
                    );
                }
                else if (sampler_name != "pseudo_n2_linear")
                {
                    FatalIOErrorInFunction(dict)
                        << "Unknown spatialSampler " << sampler_name
                        << ", use pseudo_n2_linear or rbf" << exit(FatalIOError);
                }
            }
            
            this->readValueEntry(dict, IOobjectOption::MUST_READ);
            if (this->readMixedEntries(dict))
            {
                // Full restart
            }
            else
            {
                // Start from user entered data. Assume fixedValue.
                refValue() = *this;
                refGrad() = Zero;
                valueFraction() = 1.0;
            }
        }

        //- Construct as copy setting internal field reference
        CoupledBoundary
        (
            const CoupledBoundary& wtcsf,
            const DimensionedField<scalar, volMesh>& iF
        )
        :
            mixedFvPatchScalarField(wtcsf, iF),
            temperatureCoupledBase(patch(), wtcsf)
        {}

        //- Construct as copy setting internal field reference
        CoupledBoundary
        (
            const CoupledBoundary& wtcsf
        )
        :
            mixedFvPatchScalarField(wtcsf),
            temperatureCoupledBase(patch(), wtcsf)
        {}

        //- Return a clone
        virtual tmp<fvPatchField<scalar>> clone() const
        {
            return fvPatchField<scalar>::Clone(*this);
        }

        //- Clone with an internal field reference
        virtual tmp<fvPatchField<scalar>> clone
        (
            const DimensionedField<scalar, volMesh>& iF
        ) const
        {
            return fvPatchField<scalar>::Clone(*this, iF);
        }


    // Member Functions

        // Mapping functions

        //- Map (and resize as needed) from self given a mapping object
        void autoMap
        (
            const fvPatchFieldMapper& mapper
        )
        {
            mixedFvPatchScalarField::autoMap(mapper);
            temperatureCoupledBase::autoMap(mapper);
            //mappedPatchFieldBase<scalar>::autoMap(mapper);
        }

        //- Reverse map the given fvPatchField onto this fvPatchField
        void rmap
        (
            const fvPatchField<scalar>& ptf,
            const labelList& addr
        )
        {
            mixedFvPatchScalarField::rmap(ptf, addr);

            const CoupledBoundary& tiptf =
                refCast
                <
                    const CoupledBoundary
                >(ptf);

            temperatureCoupledBase::rmap(tiptf, addr);
            //mappedPatchFieldBase<scalar>::rmap(ptf, addr);
        }

        double round_to
        (
            double value, 
            int decimal_places
        )
        {
            double scale = std::pow(10.0, decimal_places);
            return std::round(value * scale) / scale;
        }

        void push
        (
            scalar x, 
            scalar y, 
            scalar z, 
            scalar value,
            std::string push_key

        )
        {
            push_point[0] = round_to(x, 8);
            push_point[1] = round_to(y, 8);
            push_point[2] = round_to(z, 8);
            
            mui_ifs[0]->push(push_key, push_point, value);
        }

        void commit()
        {
            mui_ifs[0]->commit( iteration );
        }

        scalar fetch
        (
            scalar x, 
            scalar y, 
            scalar z, 
            std::string fetch_key
        )
        {
            fetch_point[0] = round_to(x, 8);
            fetch_point[1] = round_to(y, 8);
            fetch_point[2] = round_to(z, 8);

            scalar fetch_val = rbf_sampler
                ? mui_ifs[0]->fetch(fetch_key, fetch_point, iteration, *rbf_sampler, temporal_sampler)
                : mui_ifs[0]->fetch(fetch_key, fetch_point, iteration, spatial_sampler, temporal_sampler);

            return fetch_val;
        }

        //- Given patch temperature calculate corresponding K field. Override
        //- temperatureCoupledBase::kappa to includes effect of any
        //- explicit kappaThickness
        tmp<scalarField> kappa(const scalarField& Tp) const
        {
            // Get kappa from relevant thermo
            tmp<scalarField> tk(temperatureCoupledBase::kappa(Tp));
            return tk;
        }

        //- Update the coefficients associated with the patch field
        virtual void updateCoeffs()
        {
            mui_ifs[0]->forget(iteration-1);
            iteration += 1;

            // Parent constructor
            mixedFvPatchScalarField::updateCoeffs();    
        }


        //- Write
        void write(Ostream& os) const
        {
            mixedFvPatchField<scalar>::write(os);
            temperatureCoupledBase::write(os);
            if (sampler_name == "rbf")
            {
                os.writeEntry("spatialSampler", sampler_name);
                os.writeEntry("rbfRadius", rbf_radius);
                os.writeEntry("rbfBasis", rbf_basis);
            }
        }

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
} // End namespace Foam


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
