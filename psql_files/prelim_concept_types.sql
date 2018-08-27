set schema 'annotation';

drop table if exists prelim_concept_types;

create table prelim_concept_types as (
	select
		conceptid
		,concept_type
	from (
	select
		conceptid
		,concept_type
	from (
		select 
			subtypeid as conceptid, 
			case 
				when supertypeid = '226077000' then 'treatment' --therapeutic diet
				when supertypeid = '105958000' then 'treatment'
				when supertypeid = '373873005' then 'treatment' -- pharmaceutical / biologic product
				when supertypeid = '417176000' then 'treatment' -- growth substance
				when supertypeid = '17948008' then 'treatment' -- hematopoietic factor
				when supertypeid = '106205006' then 'treatment' -- hemostasis related substance
				when supertypeid = '373244000' then 'treatment' -- immunologic substance ? cause
				when supertypeid = '85860001' then 'treatment' -- nervous system hormone-like substance
				when supertypeid = '35069000' then 'treatment' -- neurotransmitter
				when supertypeid = '226355009' then 'treatment' -- nutrients
				when supertypeid = '226355009' then 'treatment' -- nutrients
				when supertypeid = '180045004' then 'treatment' -- Amputation stump procedure
				when supertypeid = '127599007' then 'treatment' -- Application of hip spica cast
				when supertypeid = '408816000' then 'treatment' --  Artificial rupture of membranes
				when supertypeid = '238164007' then 'treatment' --  Body wall and cavity procedures
				when supertypeid = '363006003' then 'treatment' -- Cauterization by anatomic site
				when supertypeid = '363066000' then 'treatment' -- Destructive procedure by anatomic site
				when supertypeid = '363072000' then 'treatment' --  Diagnostic procedure by site
				when supertypeid = '363076002' then 'treatment' --  Diathermy procedure by body site
				when supertypeid = '447854000' then 'treatment' --  Dilation of enterostomy stoma
				when supertypeid = '75503001' then 'treatment' -- Excision of cyst of Gartner's duct
				when supertypeid = '31075001' then 'treatment' -- Excision of cyst of Müllerian duct in male 
				when supertypeid = '66136000' then 'treatment' -- Excision of Müllerian duct
				when supertypeid = '30700006' then 'treatment' -- Excision of transplanted tissue or organ
				when supertypeid = '8253009' then 'treatment' -- Excision of Wolffian duct
				when supertypeid = '66201006' then 'treatment' -- Extraction of fetus
				when supertypeid = '266783009' then 'treatment' -- Face to pubes conversion
				when supertypeid = '182656008' then 'treatment' -- General body warming therapy
				when supertypeid = '26667003' then 'treatment' -- Incision and packing of wound
				when supertypeid = '285837007' then 'treatment' -- Injection into body site
				when supertypeid = '76790002' then 'treatment' --  Insertion of tissue expander
				when supertypeid = '363186003' then 'treatment' --  Introduction of substance by body site 
				when supertypeid = '182655007' then 'treatment' --  Local heating - infrared irradiation
				when supertypeid = '363195006' then 'treatment' -- Manipulation procedure by body site
				when supertypeid = '177203002' then 'treatment' -- Manual removal of products of conception from delivered uterus
				when supertypeid = '9803001' then 'treatment' -- Medical procedure on body region
				when supertypeid = '229319000' then 'treatment' --  Mobilizing of body part 
				when supertypeid = '179986009' then 'treatment' -- Multisystem procedure 
				when supertypeid = '91097001' then 'treatment' -- Neuromuscular procedure
				when supertypeid = '448779009' then 'treatment' -- Occlusion of systemic to pulmonary artery shunt using transluminal embolic device 
				when supertypeid = '233233001' then 'treatment' -- Operation on systemic to pulmonary artery shunt 
				when supertypeid = '236994008' then 'treatment' -- Placental delivery procedure
				when supertypeid = '129152004' then 'treatment' --  Procedure on back 
				--when supertypeid = '118664000' then 'treatment' -- Procedure on body system

				when supertypeid = '56402004' then 'diagnostic' --  Biopsy of mucous membrane (procedure)
				when supertypeid = '392089008' then 'treatment' -- Breast procedure (procedure)

				when supertypeid = '303679003' then 'diagnostic' -- Computed tomography of systems (procedure)
				when supertypeid = '265244003' then 'treatment' -- Endocrine system and/or breast operations (procedure)
				when supertypeid = '284366008' then 'diagnostic' -- Examination of body system (procedure)
				when supertypeid = '303942004' then 'diagnostic' -- Fluoroscopy of systems (procedure)
				when supertypeid = '265045007' then 'treatment' -- Lung and/or mediastinum operations (procedure)
				when supertypeid = '234244004' then 'treatment' -- Lymphatic, spleen and bone marrow procedures (procedure)
				when supertypeid = '241619004' then 'diagnostic' -- Magnetic resonance imaging of thoracic inlet (procedure)
				when supertypeid = '303859008' then 'diagnostic' -- Nuclear medicine study of systems (procedure)
				when supertypeid = '108108009' then 'treatment' -- Obstetrics manipulation (procedure)
				when supertypeid = '118672003' then 'treatment' --  Procedure on cardiovascular system (procedure)
				when supertypeid = '118673008' then 'diagnostic' -- Procedure on digestive system (procedure)
				when supertypeid = '118683007' then 'diagnostic' -- Procedure on ear and related structures (procedure)
				when supertypeid = '118681009' then 'treatment' -- Procedure on endocrine system (procedure)
				when supertypeid = '118685000' then 'treatment' -- Procedure on hematopoietic system (procedure)
				when supertypeid = '118686004' then 'diagnostic' -- Procedure on immune system (procedure)
				when supertypeid = '118665004' then 'treatment' -- Procedure on integumentary system (procedure)
				when supertypeid = '118688003' then 'treatment' -- Procedure on lymphatic system (procedure)
				when supertypeid = '118696008' then 'treatment' -- Procedure on mediastinum (procedure)
				when supertypeid = '118666003' then 'treatment' -- Procedure on musculoskeletal system (procedure)
				when supertypeid = '118678004' then 'diagnostic' -- Procedure on nervous system (procedure)
				when supertypeid = '118669005' then 'treatment' -- Procedure on respiratory system (procedure)
				when supertypeid = '371560009' then 'diagnostic' -- Procedure on visual system (procedure)
				when supertypeid = '168600009' then 'diagnostic' -- Thoracic inlet X-ray (procedure)
				when supertypeid = '231345002' then 'treatment' -- Topical local anesthetic to mucous membrane (procedure)
				when supertypeid = '303911009' then 'diagnostic' -- Ultrasound studies of systems (procedure)

				when supertypeid = '118949002' then 'treatment' -- Procedure on extremity 
				when supertypeid = '118754003' then 'treatment' -- Procedure on gland 
				when supertypeid = '118950002' then 'treatment' -- Procedure on head AND/OR neck
				when supertypeid = '118717007' then 'treatment' -- Procedure on organ 
				when supertypeid = '699465002' then 'treatment' -- Procedure on region of shoulder girdle
				when supertypeid = '118738001' then 'treatment' -- Procedure on soft tissue 
				when supertypeid = '118694006' then 'treatment' -- Procedure on trunk 
				when supertypeid = '78699000' then 'treatment' -- Radical amputation 
				when supertypeid = '429685005' then 'treatment' --  Radiotherapy by body site
				when supertypeid = '405450003' then 'treatment' -- Revision of hindquarter amputation
				when supertypeid = '307991006' then 'treatment' -- Revision of mastectomy scar
				when supertypeid = '363312001' then 'treatment' -- Stimulation procedure by body site
				when supertypeid = '363320004' then 'treatment' -- Surgical repair procedure by body site 
				when supertypeid = '442460002' then 'treatment' -- procedure on wound
				when supertypeid = '386637004' then 'treatment' -- obstetric procedure
				when supertypeid = '243120004' then 'treatment' -- Regimes and therapies 
				when supertypeid = '409002' then 'treatment' -- Food allergy diet 
				when supertypeid = '410942007' then 'treatment' --  Drug or medicament (substance)
				when supertypeid = '445839007' then 'treatment' --  Insertion of nasogastric feeding tube using endoscopy for upper gastrointestinal tract guidance (procedure)
				when supertypeid = '309041002' then 'treatment' -- Operations by intention (procedure)
				when supertypeid = '362964009' then 'treatment' -- Palliative procedure (procedure)
				when supertypeid = '277132007' then 'treatment' -- Therapeutic procedure (procedure)

				when supertypeid = '169443000' then 'prevention' --  Preventive procedure (procedure)
				when supertypeid = '20135006' then 'screening' -- Screening procedure (procedure)

				when supertypeid = '386811000' then 'diagnostic' -- Fetal procedure DIAGNOSTIC
				when supertypeid = '243773009' then 'diagnostic' -- Fetal blood sampling
				when supertypeid = '371571005' then 'diagnostic' -- Imaging by body site DIAGNOSTIC
				when supertypeid = '363244004' then 'diagnostic' -- Nuclear medicine study by site
				when supertypeid = '5880005' then 'diagnostic'  --  Physical examination procedure Diagnostic
				when supertypeid = '302381002' then 'diagnostic' -- Placental biopsy  Diagnostic
				when supertypeid = '285579008' then 'diagnostic' --  Taking swab from body site

				when supertypeid = '108252007' then 'diagnostic' -- laboratory procedure
				
				when supertypeid = '104464008' then 'diagnostic' -- Acid phosphatase measurement, forensic examination (procedure)
				when supertypeid = '103693007' then 'diagnostic' -- Diagnostic procedure (procedure)
				when supertypeid = '258174001' then 'diagnostic' -- Imaging guidance procedure (procedure)
				when supertypeid = '362964009' then 'treatment' -- Palliative procedure (procedure)
				when supertypeid = '20135006' then 'diagnostic' --  Screening procedure (procedure)
				when supertypeid = '277132007' then 'treatment' -- Therapeutic procedure (procedure)


				when supertypeid = '104464008' then 'diagnostic'-- Acid phosphatase measurement, forensic examination
				when supertypeid = '432442004' then 'diagnostic' -- Collection of forensic data (procedure)
				when supertypeid = '21268002' then 'diagnostic'  -- Cytopathology procedure, forensic (procedure)
				when supertypeid = '103693007' then 'diagnostic' --  Diagnostic procedure (procedure)
				when supertypeid = '5785009' then 'diagnostic' -- Forensic autopsy (procedure)
				when supertypeid = '446185002' then 'diagnostic' --  Forensic computed tomography (procedure)
				when supertypeid = '446186001' then 'diagnostic' -- Forensic magnetic resonance imaging (procedure)
				when supertypeid = '446181006' then 'diagnostic' -- Forensic X-ray (procedure)
				when supertypeid = '10821005' then 'diagnostic' -- Gastric fluid analysis, food, forensic (procedure)
				when supertypeid = '258174001' then 'diagnostic' -- Imaging guidance procedure (procedure)
				when supertypeid = '122869004' then 'diagnostic' -- Measurement procedure (procedure)
				when supertypeid = '14766002' then 'diagnostic' -- Aspiration (procedure)
				when supertypeid = '86273004' then 'diagnostic' -- Biopsy (procedure)

				when supertypeid = '410607006' then 'cause' -- organism
				when supertypeid = '88878007' then 'cause' -- protein
				when supertypeid = '106197002' then 'cause' -- Mediator of immune response AND/OR inflammation (substance)
				when supertypeid = '106192008' then 'cause' -- Complement related substance (substance)
				when supertypeid = '7120007' then 'cause' --  Antigen (substance)

				when supertypeid = '404684003' then 'symptom' -- clinical finding
				when supertypeid = '4147007' then 'condition' -- Mass
				when supertypeid = '123037004' then 'anatomy' -- body structure

				
				when supertypeid = '363787002' then 'observable' -- observable entity

				when supertypeid = '362981000' then 'qualifier' -- qualifier value

	   			when supertypeid = '64572001' then 'condition' -- disease
	   		end as concept_type

		from snomed.curr_transitive_closure_f tr
		left outer join (select
									conceptid
									,1 as match
								from annotation.active_descriptions d
								where term like '%(finding)%' or term like '%(disorder)%') j
		on tr.subtypeid = j.conceptid
		where j.match is null
	) tb
	where concept_type is not null and conceptid not in ('182813001', '276239002')
	
	union all

	select
		conceptid
		,case 
			when term like '%(finding)%' then 'symptom'
			when term like '%(disorder)%' then 'condition' end as concept_type
		from annotation.active_descriptions

	) f 
	where concept_type is not null
);

create index ct_conceptid on prelim_concept_types(conceptid);
create index ct_concept_type on prelim_concept_types(concept_type)